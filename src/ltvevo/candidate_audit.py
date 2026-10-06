"""Conservative static audit of Agent-generated candidate source."""

from __future__ import annotations

import ast
import re


def _scope_nodes(function: ast.FunctionDef):
    def descend(node: ast.AST):
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                yield from descend(child)
    return descend(function)


def _torch_call(node: ast.AST, method: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == method and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch")


def _nonfinite_guard_variable(test: ast.expr) -> str | None:
    if not isinstance(test, ast.UnaryOp) or not isinstance(test.op, ast.Not):
        return None
    checked = test.operand
    if (isinstance(checked, ast.Call) and isinstance(checked.func, ast.Name)
            and checked.func.id == "bool" and len(checked.args) == 1):
        checked = checked.args[0]
    if (not isinstance(checked, ast.Call) or not isinstance(checked.func, ast.Attribute)
            or checked.func.attr != "all" or checked.args):
        return None
    finite = checked.func.value
    if (_torch_call(finite, "isfinite") and len(finite.args) == 1
            and isinstance(finite.args[0], ast.Name)):
        return finite.args[0].id
    return None


def _zero_replacement(statement: ast.stmt, variable: str) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    return (isinstance(target, ast.Name) and target.id == variable
            and any(_torch_call(statement.value, method)
                    for method in ("zeros", "zeros_like")))


def _returned_names(value: ast.expr) -> set[str]:
    if isinstance(value, ast.Name):
        return {value.id}
    if isinstance(value, ast.IfExp):
        return _returned_names(value.body) | _returned_names(value.orelse)
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr in {"cpu", "numpy", "detach", "clamp", "to", "float"}):
        return _returned_names(value.func.value)
    return set()


def _sanitizer(value: ast.expr) -> str | None:
    if _torch_call(value, "nan_to_num") and value.args:
        return "non-finite predictions are silently sanitized."
    if (_torch_call(value, "where") and len(value.args) >= 3
            and _torch_call(value.args[0], "isfinite")
            and len(value.args[0].args) == 1
            and isinstance(value.args[0].args[0], ast.Name)):
        variable = value.args[0].args[0].id
        normal = value.args[1]
        fallback = value.args[2]
        if (isinstance(normal, ast.Name) and normal.id == variable
                and ((isinstance(fallback, ast.Constant) and fallback.value == 0)
                     or _torch_call(fallback, "zeros")
                     or (_torch_call(fallback, "zeros_like") and fallback.args
                         and isinstance(fallback.args[0], ast.Name)
                         and fallback.args[0].id == variable))):
            return "non-finite predictions are silently replaced with zeros."
    return None


def _prediction_findings(function: ast.FunctionDef) -> list[str]:
    nodes = list(_scope_nodes(function))
    outputs = set().union(*(_returned_names(node.value) for node in nodes
                            if isinstance(node, ast.Return) and node.value is not None))
    for node in nodes:
        if isinstance(node, ast.If):
            variable = _nonfinite_guard_variable(node.test)
            if (variable in outputs
                    and any(_zero_replacement(statement, variable) for statement in node.body)):
                return [f"Line {node.lineno}: non-finite predictions are silently replaced with zeros."]
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in outputs:
                finding = _sanitizer(node.value)
                if finding:
                    return [f"Line {node.lineno}: {finding}"]
        elif isinstance(node, ast.Return) and node.value is not None:
            finding = _sanitizer(node.value)
            if finding:
                return [f"Line {node.lineno}: {finding}"]
    return []


def _output_helpers(function: ast.FunctionDef, functions: dict[str, ast.FunctionDef]) -> list[ast.FunctionDef]:
    nodes = list(_scope_nodes(function))
    outputs = set().union(*(_returned_names(node.value) for node in nodes
                            if isinstance(node, ast.Return) and node.value is not None))
    helpers = []
    for node in nodes:
        call = None
        if isinstance(node, ast.Return):
            call = node.value
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in outputs):
            call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id in functions and call.func.id != function.name):
            helpers.append(functions[call.func.id])
    return helpers


def _quantile_mask_lines(function: ast.FunctionDef) -> list[int]:
    quantile_scores: dict[str, str] = {}
    mask_lines: dict[str, int] = {}
    assignments: dict[str, list[ast.expr]] = {}
    nodes = list(_scope_nodes(function))
    node_set = set(nodes)
    parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)
               if child in node_set}

    def conditional(node: ast.AST) -> bool:
        parent = parents.get(node)
        while parent is not None and parent is not function:
            if isinstance(parent, (ast.If, ast.For, ast.While, ast.Try, ast.Match)):
                return True
            parent = parents.get(parent)
        return False

    ordered = sorted((node for node in nodes if isinstance(node, ast.Assign)),
                         key=lambda node: (node.lineno, node.col_offset))
    for assignment in ordered:
        if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
            continue
        name = assignment.targets[0].id
        value = assignment.value
        if conditional(assignment):
            assignments.setdefault(name, []).append(value)
        else:
            assignments[name] = [value]
        quantile_scores.pop(name, None)
        mask_lines.pop(name, None)
        quantile_scores = {threshold: score for threshold, score in quantile_scores.items()
                           if score != name}
        if (_torch_call(value, "quantile") and value.args
                and isinstance(value.args[0], ast.Name)):
            quantile_scores[name] = value.args[0].id
        elif (isinstance(value, ast.Compare) and len(value.ops) == 1
              and isinstance(value.ops[0], ast.GtE) and isinstance(value.left, ast.Name)
              and len(value.comparators) == 1
              and isinstance(value.comparators[0], ast.Name)
              and quantile_scores.get(value.comparators[0].id) == value.left.id):
            mask_lines[name] = assignment.lineno

    def mask_line(value: ast.expr, seen: frozenset[str] = frozenset()) -> int | None:
        if isinstance(value, ast.Name):
            if value.id in mask_lines:
                return mask_lines[value.id]
            if value.id not in seen and value.id in assignments:
                for possible in assignments[value.id]:
                    line = mask_line(possible, seen | {value.id})
                    if line is not None:
                        return line
        elif (isinstance(value, ast.Compare) and len(value.ops) == 1
              and isinstance(value.ops[0], ast.GtE) and isinstance(value.left, ast.Name)
              and len(value.comparators) == 1
              and isinstance(value.comparators[0], ast.Name)
              and quantile_scores.get(value.comparators[0].id) == value.left.id):
            return value.lineno
        elif isinstance(value, ast.IfExp):
            return mask_line(value.body, seen) or mask_line(value.orelse, seen)
        elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
            return mask_line(value.left, seen) or mask_line(value.right, seen)
        elif isinstance(value, ast.Call):
            if _torch_call(value, "where") and value.args:
                return mask_line(value.args[0], seen)
            if (isinstance(value.func, ast.Attribute)
                    and value.func.attr in {"float", "to", "clamp", "cpu", "numpy", "detach"}):
                return mask_line(value.func.value, seen)
        return None

    for node in nodes:
        if isinstance(node, ast.Return) and node.value is not None:
            line = mask_line(node.value)
            if line is not None:
                return [line]
    return []


def audit_candidate(source: str, hypothesis: str, expected_result: str) -> dict:
    tree = ast.parse(source)
    predictor = next((node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "fit_predict"), None)
    if predictor is None:
        return {"status": "unverified", "findings": []}
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    claim = f"{hypothesis} {expected_result}"
    exact_top_k = bool(re.search(r"\btop[\s-]*(?:k\b|fraction\b|\d+(?:\.\d+)?\s*%)",
                                 claim, re.I))
    findings = []
    visited = set()
    pending = [predictor]
    while pending:
        function = pending.pop()
        if id(function) in visited:
            continue
        visited.add(id(function))
        findings.extend(_prediction_findings(function))
        if exact_top_k:
            lines = _quantile_mask_lines(function)
            if lines:
                findings.append(f"Line {lines[0]}: a >= quantile cutoff can exceed the claimed top-k fraction when scores have ties.")
        pending.extend(_output_helpers(function, functions))
    return {"status": "contradicted" if findings else "unverified", "findings": findings}

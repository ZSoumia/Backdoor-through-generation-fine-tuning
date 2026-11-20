"""Enforce the repository's test-code structure conventions."""

import ast
from pathlib import Path


TEST_DIRECTORY = Path(__file__).resolve().parent


def _parent_map(tree):
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _is_inside_function(node, parents):
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return True
        parent = parents.get(parent)
    return False


def _structure_errors(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = _parent_map(tree)
    errors = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            errors.append(f"{node.lineno}: import outside module header")
        if isinstance(node, ast.Lambda):
            errors.append(f"{node.lineno}: lambda expression")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if _is_inside_function(node, parents):
                errors.append(f"{node.lineno}: nested definition '{node.name}'")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "__import__":
                errors.append(f"{node.lineno}: dynamic import")

    header_closed = False
    for index, node in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if is_docstring:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if header_closed:
                errors.append(f"{node.lineno}: module import after executable code")
        else:
            header_closed = True
    return errors


def test_all_test_modules_obey_structure_rules():
    offenders = {}
    for path in sorted(TEST_DIRECTORY.glob("*.py")):
        errors = _structure_errors(path)
        if errors:
            offenders[path.name] = errors
    assert not offenders, f"Test code violates module structure rules: {offenders}"

"""Collect the top-level definitions a set of entry points reaches across the project's modules.

Used to inline a multi-module pipeline into one notebook: starting from named roots, follow
references to same-module globals (`name`) and to sibling modules through their import aliases
(`F.name`, `from x import name`), and return each reached definition's source with the alias
prefixes rewritten to plain names. Names defined differently in two modules are reported; the
caller decides how to rename them.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Module:
    name: str
    path: Path
    text: str
    tree: ast.Module
    defs: dict = field(default_factory=dict)      # name -> node
    aliases: dict = field(default_factory=dict)   # alias -> module name ("F" -> "features")
    imported: dict = field(default_factory=dict)  # local name -> (module, name)


def load_modules(src: Path, names: dict[str, str]) -> dict[str, Module]:
    """names: module name -> relative path."""
    mods = {}
    for m, rel in names.items():
        p = src / rel
        text = p.read_text()
        tree = ast.parse(text)
        mod = Module(m, p, text, tree)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                mod.defs[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        mod.defs[t.id] = node
                    elif isinstance(t, ast.Tuple):        # `A, B = 1, 2`
                        for e in t.elts:
                            if isinstance(e, ast.Name):
                                mod.defs[e.id] = node
            elif isinstance(node, ast.Import):
                for a in node.names:
                    base = a.name.split(".")[-1]
                    if base in names:
                        mod.aliases[a.asname or base] = base
            elif isinstance(node, ast.ImportFrom) and node.module:
                base = node.module.split(".")[-1]
                if base in names:
                    for a in node.names:
                        mod.imported[a.asname or a.name] = (base, a.name)
        mods[m] = mod
    return mods


def source_of(mod: Module, node) -> str:
    lines = mod.text.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "".join(lines[start - 1:node.end_lineno]).rstrip() + "\n"


def references(mod: Module, node) -> set[tuple[str, str]]:
    """(module, name) pairs a definition refers to."""
    out = set()
    local_imports = {}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):          # function-local `import x as Y`
            for a in sub.names:
                local_imports[a.asname or a.name.split(".")[-1]] = a.name.split(".")[-1]
    aliases = {**mod.aliases, **local_imports}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                and sub.value.id in aliases:
            out.add((aliases[sub.value.id], sub.attr))
        elif isinstance(sub, ast.Name):
            if sub.id in mod.defs and mod.defs[sub.id] is not node:
                out.add((mod.name, sub.id))
            elif sub.id in mod.imported:
                out.add(mod.imported[sub.id])
    return out


def reach(mods: dict[str, Module], roots: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Reached (module, name) in dependency order (definitions before their users)."""
    seen, order = set(), []

    def visit(key):
        if key in seen:
            return
        seen.add(key)
        m, n = key
        if m not in mods or n not in mods[m].defs:
            return                                   # stdlib / third-party
        node = mods[m].defs[n]
        for dep in sorted(references(mods[m], node)):
            visit(dep)
        order.append(key)

    for r in roots:
        visit(r)
    return order


def rewrite(mods: dict[str, Module], key: tuple[str, str], rename: dict) -> str:
    """Source of a definition with `Alias.name` -> plain (possibly renamed) names."""
    m, n = key
    mod = mods[m]
    src = source_of(mod, mod.defs[n])
    node = mod.defs[n]
    local_imports = {}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            for a in sub.names:
                local_imports[a.asname or a.name.split(".")[-1]] = a.name.split(".")[-1]
    aliases = {**mod.aliases, **local_imports}
    # drop function-local imports of sibling modules
    for alias, target in local_imports.items():
        if target in mods:
            src = re.sub(rf"^\s*import {target}( as {alias})?\n", "", src, flags=re.M)
    for alias, target in aliases.items():
        src = re.sub(rf"\b{alias}\.(\w+)\b",
                     lambda g, t=target: rename.get((t, g.group(1)), g.group(1)), src)
    # same-module names that were renamed
    for (rm, rn), new in rename.items():
        if rm == m:
            src = re.sub(rf"(?<![\w.]){rn}\b", new, src)
    return src

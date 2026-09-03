"""Minimal JCAMP-DX parser for Bruker ParaVision parameter files.

Handles the subset of the format that ``visu_pars``, ``method``, ``acqp``,
``reco`` and ``subject`` actually use:

* ``##$Key=value`` scalars (numbers, enums, ``<strings>``, ``(1, 2)`` tuples)
* ``##$Key=( n[, m] )`` arrays whose values follow on continuation lines
* ``@n*(v)`` run-length shorthand inside numeric arrays
* ``<string>`` arrays and ``(a, <b>, c)`` struct arrays
* ``$$`` comment lines, which are skipped
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

_DIMS_RE = re.compile(r'^\(\s*([\d\s,]+)\)\s*$')
_REPEAT_RE = re.compile(r'@(\d+)\*\(([^)]*)\)')
_STRING_RE = re.compile(r'<([^>]*)>')
_STRUCT_RE = re.compile(r'\(([^()]*)\)')


def _to_number(token: str) -> Union[int, float, str]:
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _parse_struct_elem(token: str) -> Union[int, float, str]:
    token = token.strip()
    m = _STRING_RE.fullmatch(token)
    if m:
        return m.group(1)
    return _to_number(token)


def _parse_body(body: str, dims: List[int]) -> Any:
    body = body.strip()
    if not body:
        return [] if dims else ''

    # String array, e.g. ( 2, 65 ) <mm> <mm>, or single string ( 65 ) <RARE>
    if body.startswith('<'):
        strings = _STRING_RE.findall(body)
        if len(dims) <= 1 or (len(dims) == 2 and len(strings) == 1):
            return strings[0] if len(strings) == 1 else strings
        return strings

    # Struct array, e.g. (9, <FG_SLICE>, <>, 0, 2) (75, <FG_DIFFUSION>, ...)
    if body.startswith('('):
        return [tuple(_parse_struct_elem(t) for t in inner.split(','))
                for inner in _STRUCT_RE.findall(body)]

    # Numeric array, possibly with @n*(v) run-length tokens
    values: List[Union[int, float, str]] = []
    pos = 0
    for m in _REPEAT_RE.finditer(body):
        values.extend(_to_number(t) for t in body[pos:m.start()].split())
        count = int(m.group(1))
        values.extend([_to_number(m.group(2).strip())] * count)
        pos = m.end()
    values.extend(_to_number(t) for t in body[pos:].split())

    if dims and len(dims) > 1 and len(values) == _product(dims):
        return _reshape(values, dims)
    return values


def _product(dims: List[int]) -> int:
    out = 1
    for d in dims:
        out *= d
    return out


def _reshape(values: list, dims: List[int]) -> list:
    if len(dims) == 1:
        return list(values)
    step = _product(dims[1:])
    return [_reshape(values[i * step:(i + 1) * step], dims[1:]) for i in range(dims[0])]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    m = _STRING_RE.fullmatch(value)
    if m:
        return m.group(1)
    if value.startswith('(') and value.endswith(')'):
        return tuple(_parse_struct_elem(t) for t in value[1:-1].split(','))
    return _to_number(value)


def read_jcampdx(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse a Bruker parameter file into a ``{name: value}`` dict.

    Keys are stored without the ``$`` prefix (``VisuCoreSize`` not
    ``$VisuCoreSize``). Non-``$`` header keys like ``TITLE`` are kept too.
    """
    text = Path(path).read_text(errors='replace').replace('\r\n', '\n')
    params: Dict[str, Any] = {}

    records = re.split(r'\n(?=##)', text)
    for record in records:
        if not record.startswith('##'):
            continue
        header, _, rest = record.partition('\n')
        key, sep, value = header[2:].partition('=')
        if not sep:
            continue
        key = key.strip().lstrip('$')
        body_lines = [ln for ln in rest.split('\n') if not ln.startswith('$$')]
        body = '\n'.join(body_lines)

        m = _DIMS_RE.match(value.strip())
        if m and body.strip():
            dims = [int(d) for d in m.group(1).replace(',', ' ').split()]
            params[key] = _parse_body(body, dims)
        else:
            scalar = value.strip()
            if body.strip():
                scalar = scalar + ' ' + body.strip()
            params[key] = _parse_scalar(scalar)
    return params

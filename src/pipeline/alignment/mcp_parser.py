"""Parse Match Charting Project shot strings into structured shot tokens.

MCP CSV format: one row per point. The '1st' column holds the shot string for
the first-serve rally; '2nd' holds the second-serve rally when the first serve
faulted. Each shot string is a compact sequence of serve codes, shot-type
letters, direction/depth digits, and outcome codes.

Shot string encoding (from MCP MatchChart instructions):
  Serve placement: 4=wide, 5=body, 6=T
  Shot types: f=forehand, b=backhand, r=forehand_slice, s=backhand_slice,
              v=volley, z=backhand_volley, l=lob, o=overhead,
              j=forehand_swing_volley, k=backhand_swing_volley,
              y=tweener, p=forehand_drop, q=backhand_drop
  Modifiers (optional, between type and direction): + = approach, - = at net,
              = = baseline, ^ = overhead position
  Direction: 1=DTL, 2=CC/middle, 3=opposite-DTL
  Depth (optional, after direction): 7=shallow, 8=mid, 9=deep
  Outcome (terminal, last shot only): * = winner, @ = unforced error,
              # = forced error, n = net, w = wide, d = deep, x = wide+deep

Rally length convention (MCP): all shots including serve, excluding the shot
that resulted in an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


_SHOT_LETTERS = frozenset("fbrsvlozkjyutpq")
_MODIFIERS = frozenset("=+-^")
_DIRECTION = frozenset("123")
_DEPTH = frozenset("789")
_OUTCOME = frozenset("*@#nwdx")
_SERVE_DIGITS = frozenset("456789")

SHOT_TYPE_NAMES: dict[str, str] = {
    "f": "forehand",
    "b": "backhand",
    "r": "forehand_slice",
    "s": "backhand_slice",
    "v": "volley",
    "z": "backhand_volley",
    "l": "lob",
    "o": "overhead",
    "j": "forehand_swing_volley",
    "k": "backhand_swing_volley",
    "y": "tweener",
    "p": "forehand_drop",
    "q": "backhand_drop",
    "serve": "serve",
}

DIRECTION_NAMES: dict[int, str] = {1: "DTL", 2: "CC", 3: "middle"}
DEPTH_NAMES: dict[int, str] = {7: "shallow", 8: "mid", 9: "deep"}
SERVE_PLACEMENT_NAMES: dict[int, str] = {4: "wide", 5: "body", 6: "T"}


@dataclass
class ShotToken:
    """One shot event parsed from an MCP shot string."""

    is_serve: bool
    shot_type: str              # e.g. 'f', 'b', 'serve'
    direction: Optional[int]    # 1, 2, 3 (None for aces/double-faults)
    depth: Optional[int]        # 7, 8, 9 (None if not recorded)
    serve_placement: Optional[int] = None   # 4–6 for serves
    modifiers: str = ""         # e.g. '+', '-='
    outcome: Optional[str] = None           # e.g. '*', 'n#', 'd@'


@dataclass
class MCPPoint:
    """One point from the Match Charting Project CSV."""

    match_id: str
    pt: int
    set1: int
    set2: int
    gm1: int
    gm2: int
    pts: str        # e.g. '40-30'
    svr: int        # 1 or 2
    pt_winner: int  # 1 or 2
    shots: list[ShotToken] = field(default_factory=list)
    rally_len: int = 0          # shots including serve, excluding error shot
    is_second_serve: bool = False


def parse_shot_string(s: str) -> list[ShotToken]:
    """Tokenise an MCP shot string into a list of ShotTokens.

    Handles both first-serve and second-serve strings.  The semicolon `;`
    used by some contributors as a visual separator is stripped before
    parsing.

    Example:
        parse_shot_string("4b38f1r2f-1l1o1*")
        → [ShotToken(serve,placement=4), ShotToken(b,dir=3,depth=8),
           ShotToken(f,dir=1), ShotToken(r,dir=2), ShotToken(f,dir=1,mod='-'),
           ShotToken(l,dir=1), ShotToken(o,dir=1,outcome='*')]
    """
    s = s.strip().replace(";", "")
    tokens: list[ShotToken] = []
    i = 0
    n = len(s)
    first_token = True

    while i < n:
        c = s[i]

        # Serve: first character is always a serve placement digit.
        if first_token and c in _SERVE_DIGITS:
            placement = int(c)
            i += 1
            outcome = _consume_outcome(s, i, n)
            i += len(outcome)
            tokens.append(ShotToken(
                is_serve=True,
                shot_type="serve",
                direction=None,
                depth=None,
                serve_placement=placement,
                outcome=outcome or None,
            ))
            first_token = False
            continue

        # Shot: starts with a shot-type letter.
        if c in _SHOT_LETTERS:
            shot_type = _consume_letters(s, i, n)
            i += len(shot_type)
            modifiers = _consume_modifiers(s, i, n)
            i += len(modifiers)
            direction: Optional[int] = None
            if i < n and s[i] in _DIRECTION:
                direction = int(s[i])
                i += 1
            depth: Optional[int] = None
            if i < n and s[i] in _DEPTH:
                depth = int(s[i])
                i += 1
            outcome = _consume_outcome(s, i, n)
            i += len(outcome)
            tokens.append(ShotToken(
                is_serve=False,
                shot_type=shot_type,
                direction=direction,
                depth=depth,
                modifiers=modifiers,
                outcome=outcome or None,
            ))
            first_token = False
            continue

        # Unknown character — skip.
        i += 1

    return tokens


def _consume_letters(s: str, i: int, n: int) -> str:
    result = ""
    while i < n and s[i] in _SHOT_LETTERS:
        result += s[i]
        i += 1
    return result


def _consume_modifiers(s: str, i: int, n: int) -> str:
    result = ""
    while i < n and s[i] in _MODIFIERS:
        result += s[i]
        i += 1
    return result


def _consume_outcome(s: str, i: int, n: int) -> str:
    result = ""
    while i < n and s[i] in _OUTCOME:
        result += s[i]
        i += 1
    return result


def _rally_len(shots: list[ShotToken]) -> int:
    """Rally length per MCP convention: all shots except the error shot."""
    if not shots:
        return 0
    last_outcome = shots[-1].outcome or ""
    is_error = bool(last_outcome) and "*" not in last_outcome
    return len(shots) - (1 if is_error else 0)


def load_mcp_csv(path: str | Path) -> list[MCPPoint]:
    """Load an MCP points CSV and return a parsed MCPPoint per row.

    Handles both first-serve and second-serve points.  When the '2nd' column
    is non-empty the second-serve string is used as the rally; when it is
    empty the '1st' string is used.
    """
    df = pd.read_csv(path, keep_default_na=False)
    points: list[MCPPoint] = []

    for _, row in df.iterrows():
        first_str = str(row.get("1st", "")).strip()
        second_str = str(row.get("2nd", "")).strip()

        if second_str:
            rally_str = second_str
            is_second = True
        else:
            rally_str = first_str
            is_second = False

        shots = parse_shot_string(rally_str) if rally_str else []

        try:
            pt_winner = int(row.get("PtWinner", 0))
        except (ValueError, TypeError):
            pt_winner = 0

        points.append(MCPPoint(
            match_id=str(row["match_id"]),
            pt=int(row["Pt"]),
            set1=int(row.get("Set1", 0)),
            set2=int(row.get("Set2", 0)),
            gm1=int(row.get("Gm1", 0)),
            gm2=int(row.get("Gm2", 0)),
            pts=str(row.get("Pts", "")),
            svr=int(row.get("Svr", 1)),
            pt_winner=pt_winner,
            shots=shots,
            rally_len=_rally_len(shots),
            is_second_serve=is_second,
        ))

    return points

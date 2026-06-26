"""Tests for MCP shot string parser."""

import io
import textwrap

import pytest

from src.pipeline.alignment.mcp_parser import (
    MCPPoint,
    ShotToken,
    _rally_len,
    load_mcp_csv,
    parse_shot_string,
)


# ─────────────────────────────── parse_shot_string ──────────────────────────

def test_ace():
    tokens = parse_shot_string("4*")
    assert len(tokens) == 1
    t = tokens[0]
    assert t.is_serve and t.shot_type == "serve"
    assert t.serve_placement == 4
    assert t.outcome == "*"


def test_double_fault():
    tokens = parse_shot_string("4n")
    assert len(tokens) == 1
    assert tokens[0].is_serve
    assert tokens[0].outcome == "n"


def test_serve_plus_return_error():
    # serve wide, backhand CC unforced error
    tokens = parse_shot_string("4b2@")
    assert len(tokens) == 2
    serve, ret = tokens
    assert serve.is_serve and serve.serve_placement == 4
    assert not ret.is_serve
    assert ret.shot_type == "b"
    assert ret.direction == 2
    assert ret.outcome == "@"


def test_full_rally():
    # from MCP docs: 4b38f1r2f-1l1o1*
    tokens = parse_shot_string("4b38f1r2f-1l1o1*")
    assert tokens[0].is_serve and tokens[0].serve_placement == 4
    assert tokens[1].shot_type == "b" and tokens[1].direction == 3 and tokens[1].depth == 8
    assert tokens[2].shot_type == "f" and tokens[2].direction == 1
    assert tokens[3].shot_type == "r" and tokens[3].direction == 2
    assert tokens[4].shot_type == "f" and tokens[4].modifiers == "-" and tokens[4].direction == 1
    assert tokens[5].shot_type == "l" and tokens[5].direction == 1
    assert tokens[6].shot_type == "o" and tokens[6].direction == 1 and tokens[6].outcome == "*"


def test_modifier_approach():
    tokens = parse_shot_string("4f+2")
    assert tokens[1].modifiers == "+"
    assert tokens[1].direction == 2


def test_depth_digit_parsed():
    tokens = parse_shot_string("4f19")
    assert tokens[1].depth == 9


def test_semicolon_stripped():
    tokens_plain = parse_shot_string("4f2b1*")
    tokens_semi  = parse_shot_string("4f2;b1*")
    assert len(tokens_plain) == len(tokens_semi)
    for a, b in zip(tokens_plain, tokens_semi):
        assert a.shot_type == b.shot_type


def test_empty_string():
    assert parse_shot_string("") == []
    assert parse_shot_string("   ") == []


def test_unknown_characters_skipped():
    # '!' is not a valid token; parser should skip and still find the serve
    tokens = parse_shot_string("!4*")
    assert len(tokens) == 1
    assert tokens[0].is_serve


# ─────────────────────────────── _rally_len ─────────────────────────────────

def test_rally_len_ace():
    tokens = parse_shot_string("4*")
    assert _rally_len(tokens) == 1


def test_rally_len_serve_return_winner():
    # serve + return winner: both count
    tokens = parse_shot_string("4f2*")
    assert _rally_len(tokens) == 2


def test_rally_len_serve_return_error():
    # serve + return unforced error: error shot excluded
    tokens = parse_shot_string("4f2@")
    assert _rally_len(tokens) == 1


def test_rally_len_empty():
    assert _rally_len([]) == 0


# ─────────────────────────────── load_mcp_csv ───────────────────────────────

_MINIMAL_CSV = textwrap.dedent("""\
    match_id,Pt,Set1,Set2,Gm1,Gm2,Pts,Svr,1st,2nd,PtWinner
    TEST_MATCH,1,0,0,0,0,0-0,1,4f2*,,1
    TEST_MATCH,2,0,0,0,0,15-0,2,,4b3@,2
""")


def test_load_mcp_csv_basic(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text(_MINIMAL_CSV)
    pts = load_mcp_csv(p)
    assert len(pts) == 2


def test_load_mcp_csv_first_serve(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text(_MINIMAL_CSV)
    pt = load_mcp_csv(p)[0]
    assert pt.match_id == "TEST_MATCH"
    assert pt.svr == 1
    assert not pt.is_second_serve
    assert pt.shots[0].is_serve
    assert pt.rally_len == 2  # serve + winner


def test_load_mcp_csv_second_serve(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text(_MINIMAL_CSV)
    pt = load_mcp_csv(p)[1]
    assert pt.is_second_serve
    assert pt.shots[1].shot_type == "b"
    assert pt.rally_len == 1  # serve only (return error excluded)

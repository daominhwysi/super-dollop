"""
Unit tests for VSL unified CLI entrypoint (sequence_labelling/cli.py).
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch
import pytest

from sequence_labelling.cli import build_parser, main


def test_cli_parser_registered_subcommands():
    parser = build_parser()
    subparsers_action = [
        action for action in parser._actions
        if action.dest == "subcommand"
    ][0]
    registered = list(subparsers_action.choices.keys())
    expected = [
        "paths",
        "ocr",
        "cartography",
        "metadata",
        "parse",
        "audit",
        "repair",
        "branch",
        "dataset",
        "evaluate",
        "status",
        "tmux",
        "monitor",
    ]
    for sub in expected:
        assert sub in registered, f"Missing subcommand: {sub}"


def test_cli_paths_command(capsys):
    with patch.object(sys, "argv", ["vsl", "paths"]):
        main()
    captured = capsys.readouterr()
    assert "workspace=" in captured.out
    assert "data=" in captured.out
    assert "models=" in captured.out


def test_cli_status_command(capsys):
    with patch.object(sys, "argv", ["vsl", "status"]):
        main()
    captured = capsys.readouterr()
    assert "VSL PIPELINE STATUS DASHBOARD" in captured.out
    assert "[1. OCR Pipeline]" in captured.out
    assert "[2. Sequence Labelling Annotation Pipeline]" in captured.out
    assert "[3. Quality Review & Audit Status]" in captured.out


def test_cli_tmux_list_command(capsys):
    with patch.object(sys, "argv", ["vsl", "tmux", "list"]):
        main()
    captured = capsys.readouterr()
    assert "TMUX" in captured.out

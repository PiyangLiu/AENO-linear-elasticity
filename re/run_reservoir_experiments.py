#!/usr/bin/env python3
"""Stable entry point for the reservoir AENO ablation and seed suite."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    suite = Path(__file__).resolve().parent / "ab" / "123.py"
    runpy.run_path(str(suite), run_name="__main__")

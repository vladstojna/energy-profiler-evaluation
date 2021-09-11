#!/usr/bin/env python3

import argparse
import csv
import sys
from typing import Any, Optional


def read_from(path: Optional[str]) -> Any:
    return sys.stdin if not path else open(path, "r")


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "source_file",
        action="store",
        help="file to convert (default: stdin)",
        nargs="?",
        type=str,
        default=None,
    )
    return parser


def main():
    def not_empty_not_comment(f):
        for r in f:
            row = r.strip()
            if row and not row.startswith("#"):
                yield row

    parser = argparse.ArgumentParser(
        description="Extract first timestamp from timeprinter output"
    )
    args = add_arguments(parser).parse_args()
    with read_from(args.source_file) as f:
        csvrdr = csv.DictReader(not_empty_not_comment(f))
        if not csvrdr.fieldnames:
            raise AssertionError("File has no fieldnames")
        first_row = next(iter(csvrdr), None)
        if first_row is None:
            raise AssertionError("File has no data rows")
        time = first_row.get("time")
        if time is None:
            raise AssertionError("No 'time' column")
        print(time)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import sys
import csv
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def read_from(path: Optional[str]) -> Any:
    return sys.stdin if not path else open(path, "r")


def output_to(path: Optional[str]) -> Any:
    return sys.stdout if not path else open(path, "w")


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    def positive_int_or_float(s: str) -> Union[int, float]:
        try:
            val = int(s)
            if val <= 0:
                raise argparse.ArgumentTypeError("value must be positive")
            return val
        except ValueError:
            try:
                val = float(s)
                if val <= 0:
                    raise argparse.ArgumentTypeError("value must be positive")
            except ValueError as err:
                raise argparse.ArgumentTypeError(
                    err.args[0] if len(err.args) else "could not convert value to float"
                )

    parser.add_argument(
        "source_file",
        action="store",
        help="file to convert (default: stdin)",
        nargs="?",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        help="destination file (default: stdout)",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--start",
        action="store",
        help="absolute time of start (default: 0)",
        required=False,
        type=positive_int_or_float,
        default=0,
    )
    return parser


def main():
    def not_empty_not_comment(f):
        for r in f:
            row = r.strip()
            if row and not row.startswith("#"):
                yield row

    def int_or_float(s: str):
        try:
            return int(s)
        except ValueError:
            return float(s)

    def convert_filter_row(row: List, fieldnames: Dict[str, Tuple[int, Callable]]):
        return (
            call(ix) if not isinstance(ix, int) else call(row[ix]) if row[ix] else 0
            for _, (ix, call) in fieldnames.items()
        )

    parser = argparse.ArgumentParser(
        description="Convert perf stat output to a more plottable format"
    )
    args = add_arguments(parser).parse_args()
    with read_from(args.source_file) as f:
        csvrdr = csv.reader(not_empty_not_comment(f))
        first_row = next(iter(csvrdr), None)
        if first_row:

            def inc_count(c: List[int]) -> int:
                c[0] += 1
                return c[0]

            units_meta = ["#units", "energy=J", "power=W", "time=ns"]
            noop = lambda x: x
            fieldnames = {
                "count": ([0], inc_count),
                "time": (0, lambda t: int(int_or_float(t) * 1e9) + args.start),
                first_row[3]: (1, noop),
                "counter_run_time": (4, noop),
                "counter_run_percent": (5, noop),
            }
            first_data_row = (0, 0 + args.start, 0.0, 0, 0.0)
            assert len(first_data_row) == len(fieldnames)
            with output_to(args.output) as of:
                writer = csv.writer(of)
                writer.writerow(units_meta)
                writer.writerow(fieldnames)
                writer.writerow(first_data_row)
                writer.writerow(convert_filter_row(first_row, fieldnames))
                for row in csvrdr:
                    writer.writerow(convert_filter_row(row, fieldnames))


if __name__ == "__main__":
    main()

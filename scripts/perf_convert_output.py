#!/usr/bin/env python3

import sys
import csv
import argparse
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def log(*args: Any) -> None:
    print("{}:".format(sys.argv[0]), *args, file=sys.stderr)


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
        help="absolute start time (default: 0)",
        required=False,
        type=positive_int_or_float,
        default=0,
    )
    parser.add_argument(
        "-e",
        "--end",
        action="store",
        help="absolute end time (default: 0)",
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
            for _, (_, ix, call) in fieldnames.items()
        )

    def shift_row_time(row: List, shift_by: int, time_ix: int):
        return (
            row[ix] if ix != time_ix else row[ix] - shift_by for ix in range(len(row))
        )

    parser = argparse.ArgumentParser(
        description="Convert perf stat output to a more plottable format"
    )
    args = add_arguments(parser).parse_args()
    if args.start and args.end and args.end <= args.start:
        raise parser.error("-e/--end must be greater than -s/--start")
    if args.end and not args.start:
        raise parser.error("-e/--end requires -s/--start")
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
                "count": (0, [0], inc_count),
                "time": (1, 0, lambda t: int(int_or_float(t) * 1e9) + args.start),
                first_row[3]: (2, 1, noop),
                "counter_run_time": (3, 4, noop),
                "counter_run_percent": (4, 5, noop),
            }
            first_data_row = [0, 0 + args.start, "0.0", "0", "0.0"]
            assert len(first_data_row) == len(fieldnames)
            with output_to(args.output) as of:
                writer = csv.writer(of)
                writer.writerow(units_meta)
                writer.writerow(fieldnames)

                if not args.end:
                    writer.writerow(first_data_row)
                    writer.writerow(convert_filter_row(first_row, fieldnames))
                    for row in csvrdr:
                        writer.writerow(convert_filter_row(row, fieldnames))
                else:
                    data = [first_data_row]
                    data += [
                        [c for c in convert_filter_row(r, fieldnames)]
                        for r in itertools.chain((first_row,), csvrdr)
                    ]
                    time_ix = fieldnames["time"][0]
                    last_time = data[-1][time_ix]
                    if last_time > args.end:
                        shift_by = last_time - args.end
                        log("perf overhead ~", shift_by, "ns")
                        log("shifting time values left by", shift_by, "ns")
                        for row in data:
                            writer.writerow(shift_row_time(row, shift_by, time_ix))
                    else:
                        log("provided end time >= {}".format(last_time))
                        for row in data:
                            writer.writerow(row)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import sys
from typing import Any, Dict, Iterable, Optional, Tuple, Union


def read_from(path: Optional[str]) -> Any:
    return sys.stdin if not path else open(path, "r")


def output_to(path: Optional[str]) -> Any:
    return sys.stdout if not path else open(path, "w")


def int_or_float(s: str) -> Union[int, float]:
    try:
        return int(s)
    except ValueError:
        return float(s)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    def _int_or_float(s: str) -> Union[int, float]:
        try:
            return int_or_float(s)
        except ValueError as err:
            raise argparse.ArgumentTypeError(
                err.args[0] if err.args else "could not convert value to float"
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
        "-a",
        "--amount",
        action="store",
        help="subtract AMOUNT from all time values (default: first value)",
        required=False,
        type=_int_or_float,
        default=None,
        metavar="AMOUNT",
    )
    return parser


def read_input_file(f: Iterable) -> Tuple[csv.reader, csv.DictReader]:
    meta, fieldnames, data = [], [], []
    is_fieldnames = True
    for row in f:
        if row.strip().startswith("#"):
            meta.append(row)
        elif is_fieldnames:
            fieldnames.append(row)
            is_fieldnames = False
        else:
            data.append(row)
    if not fieldnames:
        raise AssertionError("File has no fieldnames")
    if not data:
        raise AssertionError("File has no data rows")
    return csv.reader(meta), csv.DictReader(data, next(iter(csv.reader(fieldnames))))


def main():
    def filter_row(row: Dict, amount: Union[int, float]):
        return (
            (k, v if k != "time" else int_or_float(v) - amount) for k, v in row.items()
        )

    parser = argparse.ArgumentParser(
        description="Transform time values from absolute to relative"
    )
    args = add_arguments(parser).parse_args()
    with read_from(args.source_file) as f:
        meta, data = read_input_file(f)
        with output_to(args.output) as of:
            csv.writer(of).writerows(meta)
            writer = csv.DictWriter(of, data.fieldnames)
            writer.writeheader()
            first_row = next(iter(data), None)
            if first_row is not None:
                if args.amount is None and "time" in first_row:
                    args.amount = int_or_float(first_row["time"])
                writer.writerow(dict(filter_row(first_row, args.amount)))
                for row in data:
                    writer.writerow(dict(filter_row(row, args.amount)))


if __name__ == "__main__":
    main()

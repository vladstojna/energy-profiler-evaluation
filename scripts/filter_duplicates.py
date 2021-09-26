#!/usr/bin/env python3

import sys
import csv
import argparse
import re
from typing import Any, Iterable, Optional, Sequence, Set, Tuple, Union


def log(*args: Any) -> None:
    print("{}:".format(sys.argv[0]), *args, file=sys.stderr)


def read_from(path: Optional[str]) -> Any:
    return sys.stdin if not path else open(path, "r")


def output_to(path: Optional[str]) -> Any:
    return sys.stdout if not path else open(path, "w")


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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


def filterable_fields(fieldnames: Sequence, pattern: str) -> Set:
    prog = re.compile(pattern)
    return {f for f in fieldnames if prog.match(f)}


def float_or_int(s: str) -> Union[float, str]:
    try:
        return int(s)
    except ValueError:
        return float(s)


def main():
    parser = argparse.ArgumentParser(
        description="Filter duplicate readings (zero energy) from a converted perf stat output"
    )
    args = add_arguments(parser).parse_args()
    with read_from(args.source_file) as f:
        meta, data = read_input_file(f)
        ffields = filterable_fields(data.fieldnames, r".+/.+/$")
        with output_to(args.output) as of:
            csv.writer(of).writerows(meta)
            dwriter = csv.DictWriter(of, data.fieldnames)
            dwriter.writeheader()
            first_row = next(iter(data), None)
            if first_row is None:
                raise AssertionError("File has no data rows")
            dwriter.writerow(first_row)
            # Add one because the count starts from the next line
            # Add two because both the fieldname and first data lines are not considered
            for ln, row in enumerate(data, start=meta.line_num + 2 + 1):
                if all(
                    float_or_int(val)
                    for val in (v for k, v in row.items() if k in ffields)
                ):
                    dwriter.writerow(row)
                else:
                    log("filtered out line", ln)


if __name__ == "__main__":
    main()

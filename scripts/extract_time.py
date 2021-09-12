#!/usr/bin/env python3

import argparse
import csv
import sys
from typing import Any, Dict, List, Optional, Sequence


class store_count(argparse.Action):
    choices = ("first", "last")
    choices_str = "{{{}}}".format(",".join(choices))
    default = 0
    metavar = "{} or COUNT".format(choices_str)

    def __init__(self, option_strings: Sequence[str], **kwargs) -> None:
        super().__init__(option_strings, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string,
    ) -> None:
        try:
            try:
                intval = int(values)
                if intval < 0:
                    raise OverflowError("count must be >= 0")
                values = intval
            except ValueError as err:
                if values not in store_count.choices:
                    raise ValueError(
                        "choice {} not in {}".format(values, store_count.choices_str)
                    )
                value = store_count.choices[store_count.choices.index(values)]
                if value == store_count.choices[0]:
                    values = 0
                elif value == store_count.choices[1]:
                    values = -1
            setattr(namespace, self.dest, values)
        except (
            OverflowError,
            ValueError,
            TypeError,
            argparse.ArgumentTypeError,
        ) as err:
            raise argparse.ArgumentError(self, err.args[0] if err.args else "<empty>")


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
    parser.add_argument(
        "-c",
        "--count",
        action=store_count,
        help="extract time with count COUNT (default: {})".format(store_count.default),
        required=False,
        default=store_count.default,
        metavar=store_count.metavar,
    )
    return parser


def get_count(row: Dict[str, str]) -> str:
    ccount = row.get("count")
    if ccount is None:
        raise AssertionError("No 'count' column")
    return ccount


def get_time(row: Dict[str, str]) -> str:
    ctime = row.get("time")
    if ctime is None:
        raise AssertionError("No 'time' column")
    return ctime


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
        data: List = list(csvrdr)
        if not data:
            raise AssertionError("File has no data rows")
        if args.count == -1 or args.count == 0:
            print(get_time(data[args.count]))
        else:
            found = False
            for row in data:
                ccount = get_count(row)
                if int(ccount) == args.count:
                    found = True
                    print(get_time(row))
            if not found:
                print(
                    "{}: time with count {} not found".format(sys.argv[0], args.count),
                    file=sys.stderr,
                )
                print(0)


if __name__ == "__main__":
    main()

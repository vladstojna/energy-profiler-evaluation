#!/usr/bin/env bash

source $(dirname "$0")/common_defs.sh

function usage
{
    local h="[-h]"
    local w="[-w <$(usage_w_string '|')>]"
    local p="[-p <profiler path>]"
    local o="[-o <output dir>]"
    local c="[-c <config dir>]"
    local i="[-i <#>]"
    echo "Usage: $0 $h $w $p $o $c $i"
    exit "$1"
}

while getopts "hw:p:c:o:i:" opt
do
    case $opt in
        p)
            prof="${OPTARG}"
            ;;
        w)
            what="${OPTARG}"
            ! is_valid_work "$what" && echoerr "Invalid work type: $what" && usage 1
            ;;
        o)
            outdir="${OPTARG}"
            ;;
        c)
            configs="${OPTARG}"
            ;;
        i)
            iters="${OPTARG}"
            ;;
        h | *)
            usage 0
            ;;
    esac
done

if [[ -z "$outdir" ]]; then
    echoerr "Option -o must be provided"
    usage 1
fi
if [[ ! -d "$outdir" ]]; then
    echoerr "$outdir does not exist"
    exit 1
fi

if [[ -z "$configs" ]]; then
    echoerr "Option -c must be provided"
    usage 1
fi
if [[ ! -d "$configs" ]]; then
    echoerr "$configs does not exist"
    exit 1
fi

if [[ -z "$prof" ]]; then
    if [[ -z "$ENERGY_PROFILER_BIN" ]]; then
        echoerr "Option -p not provided and environment variable ENERGY_PROFILER_BIN not set"
        usage 1
    fi
    prof="$ENERGY_PROFILER_BIN"
fi
if [[ ! -x "$prof" ]]; then
    echoerr "$prof does not exist or is not an executable"
    exit 1
fi

if [[ -z "$what" ]]; then
    what="$default_work"
fi
if [[ -z "$iters" ]]; then
    iters=1
fi

echo "Run: $what"
echo "Profiler: $prof"
echo "Output directory: $outdir"
echo "Config directory: $configs"
echo "Iterations: $iters"

function execute_command
{
    case "$1" in
        (sleep)
            "$prof" --no-idle -q -c "$configs/$1.xml" -o "$3.json" -- "$2" > "$3.app.csv"
            ;;
        (alternating)
            sed 's|<interval>.*</interval>|<interval>20</interval>|g' "$configs/$1.xml" | \
                "$prof" -q -o "$3.json" -- "$2" > "$3.app.csv"
            ;;
        (*)
            "$prof" -q -c "$configs/$1.xml" -o "$3.json" -- "$2" > "$3.app.csv"
            ;;
    esac
}

source $(dirname "$0")/common_loop.sh

#!/usr/bin/env bash

source $(dirname "$0")/common_defs.sh

function usage
{
    local h="[-h]"
    local w="[-w <$(usage_w_string '|')>]"
    local o="[-o <output dir>]"
    local i="[-i <#>]"
    local p="[-p <#>]"
    local e="[-e <event>]"
    echo "Usage: $0 $h $w $o $i $p $e"
    exit "$1"
}

while getopts "hw:o:i:p:" opt
do
    case $opt in
        w)
            what="${OPTARG}"
            ! is_valid_work "$what" && echoerr "Invalid work type: $what" && usage 1
            ;;
        o)
            outdir="${OPTARG}"
            ;;
        i)
            iters="${OPTARG}"
            ;;
        p)
            period="${OPTARG}"
            ;;
        e)
            event="${OPTARG}"
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

if [[ -z "$what" ]]; then
    what="$default_work"
fi
if [[ -z "$iters" ]]; then
    iters=1
fi
if [[ -z "$period" ]]; then
    period=100
fi
if [[ -z "$event" ]]; then
    event="power/energy-pkg/"
fi

echo "Run: $what"
echo "Output directory: $outdir"
echo "Iterations: $iters"
echo "Interval: $period"
echo "Event: $event"

function execute_command
{
    perf stat -o "$3.perf.csv" -a -x, -I "$period" -e "$event" "$2" > "$3.app.csv"
}

source $(dirname "$0")/common_loop.sh

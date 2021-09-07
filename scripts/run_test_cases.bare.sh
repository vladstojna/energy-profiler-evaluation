#!/usr/bin/env bash

source $(dirname "$0")/common_defs.sh

function usage
{
    local h="[-h]"
    local w="[-w <$(usage_w_string '|')>]"
    local o="[-o <output dir>]"
    local i="[-i <#>]"
    local n="[-n]"
    echo "Usage: $0 $h $w $o $i $n"
    exit "$1"
}

while getopts "hw:o:i:n" opt
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
        n)
            dry_run="true"
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

echo "Run: $what"
echo "Output directory: $outdir"
echo "Iterations: $iters"

if [[ -z "$dry_run" ]]; then
    function execute_command
    {
        $2 > "$3.app.csv"
    }
else
    function execute_command
    {
        echo ">> $2 > $3.app.csv"
    }
fi

source $(dirname "$0")/common_loop.sh

#!/usr/bin/env bash

source common_defs.sh

function usage
{
    local h="[-h]"
    local w="[-w <alternating|rng|sleep|mkl|openblas>]"
    local o="[-o <output dir>]"
    local i="[-i <#>]"
    echo "Usage: $0 $h $w $o $i"
    exit $1
}

while getopts "hw:o:i:" opt
do
    case $opt in
        w)
            what=${OPTARG}
            ! is_valid_work $what && usage 1
            ;;
        o)
            outdir=${OPTARG}
            ;;
        i)
            iters=${OPTARG}
            ;;
        h | *)
            usage 0
            ;;
    esac
done

if [[ -z $outdir ]]; then
    echoerr "Option -o must be provided"
    usage 1
fi
if [[ -z $what ]]; then
    what=$default_work
fi
if [[ -z $iters ]]; then
    iters=1
fi

echo "Run: $what"
echo "Output directory: $outdir"
echo "Iterations: $iters"

function execute_command
{
    $2 > $3.app.csv
}

source common_loop.sh

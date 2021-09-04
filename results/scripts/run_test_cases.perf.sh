#!/usr/bin/env bash

source util.sh

function usage
{
    local h="[-h]"
    local w="[-w <alternating|rng|sleep|mkl|openblas>]"
    local o="[-o <output dir>]"
    local i="[-i <#>]"
    local p="[-p <#>]"
    local e="[-e <event>]"
    echo "Usage: $0 $h $w $o $i $p $e"
    exit $1
}

while getopts "hw:o:i:p:" opt
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
        p)
            period=${OPTARG}
            ;;
        e)
            event=${OPTARG}
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
if [[ -z $period ]]; then
    period=100
fi
if [[ -z $event ]]; then
    event="power/energy-pkg/"
fi

echo "Run: $what"
echo "Output directory: $outdir"
echo "Iterations: $iters"
echo "Interval: $period"
echo "Event: $event"

for i in $(seq 0 $(($iters - 1))); do

    if is_candidate $what "sleep"; then
        cmd="$samples_dir/sleep/sleep.out 20000"
        out="$outdir/sleep-20000.$i"
        echo $cmd
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv
    fi

    if is_candidate $what "rng"; then
        export OMP_NUM_THREADS=$threads

        cmd="$samples_dir/rng/rng.out 20000000000 0.0 1.0"
        out="$outdir/rng-2e10_0_1-smton.$i"
        echo "$cmd; threads=$OMP_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        export OMP_NUM_THREADS=$cores
        out="$outdir/rng-2e10_0_1-smtoff.$i"
        echo "$cmd; threads=$OMP_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        export OMP_NUM_THREADS=1
        cmd="$samples_dir/rng/rng.out 2000000000 0.0 1.0"
        out="$outdir/rng-2e9_0_1-singlethread.$i"
        echo "$cmd; threads=$OMP_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        unset OMP_NUM_THREADS
    fi

    if is_candidate $what "alternating"; then
        export OMP_NUM_THREADS=$threads
        cmd="$samples_dir/alternating/alternating.out 200 50"
        out="$outdir/alternating-200_50-smton.$i"
        echo $cmd
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv
        unset OMP_NUM_THREADS
    fi

    if is_candidate $what "openblas"; then
        export OPENBLAS_NUM_THREADS=$cores

        pref="openblas"
        suff="smtoff.$i"

        cmd="$samples_dir/cblas/cblas-$pref.out dgemm 16000 16000 16000"
        out="$outdir/$pref-dgemm_16K_16K_16K-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/cblas/cblas-$pref.out sgemm 20000 20000 20000"
        out="$outdir/$pref-sgemm_20K_20K_20K-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-$pref.out dgesv 20000 100"
        out="$outdir/$pref-dgesv_20K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-$pref.out sgesv 26000 100"
        out="$outdir/$pref-sgesv_26K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-$pref.out dgels 10000 12000 100"
        out="$outdir/$pref-dgels_10K_12K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-$pref.out dgels 12000 10000 100"
        out="$outdir/$pref-dgels_12K_10K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-$pref.out sgels 12000 16000 100"
        out="$outdir/$pref-sgels_12K_16K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-$pref.out sgels 16000 12000 100"
        out="$outdir/$pref-sgels_16K_12K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        unset OPENBLAS_NUM_THREADS
    fi

    if is_candidate $what "mkl"; then
        pref="intel_mkl"
        suff="smtoff.$i"

        cmd="$samples_dir/cblas/cblas-intel-mkl.out dgemm 16000 16000 16000"
        out="$outdir/$pref-dgemm_16K_16K_16K-$suff"
        echo "$cmd"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/cblas/cblas-intel-mkl.out sgemm 20000 20000 20000"
        out="$outdir/$pref-sgemm_20K_20K_20K-$suff"
        echo "$cmd"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-intel-mkl.out dgesv 22000 100"
        out="$outdir/$pref-dgesv_22K_100-$suff"
        echo "$cmd"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-intel-mkl.out sgesv 28000 100"
        out="$outdir/$pref-sgesv_28K_100-$suff"
        echo "$cmd"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-intel-mkl.out dgels 15000 20000 100"
        out="$outdir/$pref-dgels_15K_20K_100-$suff"
        echo "$cmd"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-intel-mkl.out dgels 20000 15000 100"
        out="$outdir/$pref-dgels_20K_15K_100-$suff"
        echo "$cmd"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-intel-mkl.out sgels 20000 24000 100"
        out="$outdir/$pref-sgels_20K_24K_100-$suff"
        echo "$cmd"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv

        cmd="$samples_dir/lapacke/lapacke-intel-mkl.out sgels 24000 20000 100"
        out="$outdir/$pref-sgels_24K_20K_100-$suff"
        echo "$cmd"
        perf stat -o $out.perf.csv -a -x, -I $period -e $event $cmd > $out.app.csv
    fi
done

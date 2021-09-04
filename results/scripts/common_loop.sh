for i in $(seq 0 $(($iters - 1))); do

    sample_name=sleep
    if is_candidate $what $sample_name; then
        cmd="$samples_dir/$sample_name/$sample_name.out 20000"
        out="$outdir/$sample_name-20000.$i"
        echo $cmd
        execute_command "$sample_name" "$cmd" "$out"
    fi

    sample_name=rng
    if is_candidate $what $sample_name; then
        export OMP_NUM_THREADS=$threads

        cmd="$samples_dir/$sample_name/$sample_name.out 20000000000 0.0 1.0"
        out="$outdir/$sample_name-2e10_0_1-smton.$i"
        echo "$cmd; threads=$OMP_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        export OMP_NUM_THREADS=$cores
        out="$outdir/$sample_name-2e10_0_1-smtoff.$i"
        echo "$cmd; threads=$OMP_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        export OMP_NUM_THREADS=1
        cmd="$samples_dir/$sample_name/$sample_name.out 2000000000 0.0 1.0"
        out="$outdir/$sample_name-2e9_0_1-singlethread.$i"
        echo "$cmd; threads=$OMP_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        unset OMP_NUM_THREADS
    fi

    sample_name=alternating
    if is_candidate $what $sample_name; then
        export OMP_NUM_THREADS=$threads
        cmd="$samples_dir/$sample_name/$sample_name.out 200 50"
        out="$outdir/$sample_name-200_50-smton.$i"
        echo $cmd
        execute_command "$sample_name" "$cmd" "$out"
        unset OMP_NUM_THREADS
    fi

    if is_candidate $what "openblas"; then
        export OPENBLAS_NUM_THREADS=$cores

        pref="openblas"
        suff="smtoff.$i"
        sample_name=cblas

        cmd="$samples_dir/$sample_name/$sample_name-$pref.out dgemm 16000 16000 16000"
        out="$outdir/$pref-dgemm_16K_16K_16K-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-$pref.out sgemm 20000 20000 20000"
        out="$outdir/$pref-sgemm_20K_20K_20K-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        sample_name=lapacke

        cmd="$samples_dir/$sample_name/$sample_name-$pref.out dgesv 20000 100"
        out="$outdir/$pref-dgesv_20K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-$pref.out sgesv 26000 100"
        out="$outdir/$pref-sgesv_26K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-$pref.out dgels 10000 12000 100"
        out="$outdir/$pref-dgels_10K_12K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-$pref.out dgels 12000 10000 100"
        out="$outdir/$pref-dgels_12K_10K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-$pref.out sgels 12000 16000 100"
        out="$outdir/$pref-sgels_12K_16K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-$pref.out sgels 16000 12000 100"
        out="$outdir/$pref-sgels_16K_12K_100-$suff"
        echo "$cmd; threads=$OPENBLAS_NUM_THREADS"
        execute_command "$sample_name" "$cmd" "$out"

        unset OPENBLAS_NUM_THREADS
    fi

    if is_candidate $what "mkl"; then
        pref="intel_mkl"
        suff="smtoff.$i"
        sample_name=cblas

        cmd="$samples_dir/$sample_name/$sample_name-intel-mkl.out dgemm 16000 16000 16000"
        out="$outdir/$pref-dgemm_16K_16K_16K-$suff"
        echo "$cmd"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-intel-mkl.out sgemm 20000 20000 20000"
        out="$outdir/$pref-sgemm_20K_20K_20K-$suff"
        echo "$cmd"
        execute_command "$sample_name" "$cmd" "$out"

        sample_name=lapacke

        cmd="$samples_dir/$sample_name/$sample_name-intel-mkl.out dgesv 22000 100"
        out="$outdir/$pref-dgesv_22K_100-$suff"
        echo "$cmd"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-intel-mkl.out sgesv 28000 100"
        out="$outdir/$pref-sgesv_28K_100-$suff"
        echo "$cmd"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-intel-mkl.out dgels 15000 20000 100"
        out="$outdir/$pref-dgels_15K_20K_100-$suff"
        echo "$cmd"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-intel-mkl.out dgels 20000 15000 100"
        out="$outdir/$pref-dgels_20K_15K_100-$suff"
        echo "$cmd"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-intel-mkl.out sgels 20000 24000 100"
        out="$outdir/$pref-sgels_20K_24K_100-$suff"
        echo "$cmd"
        execute_command "$sample_name" "$cmd" "$out"

        cmd="$samples_dir/$sample_name/$sample_name-intel-mkl.out sgels 24000 20000 100"
        out="$outdir/$pref-sgels_24K_20K_100-$suff"
        echo "$cmd"
        execute_command "$sample_name" "$cmd" "$out"
    fi
done

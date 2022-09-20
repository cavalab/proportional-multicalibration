#!/usr/bin/bash
rdir="results/BCH/"
mkdir -p $rdir

dataset="data/bch_cleaned.csv"
methods=(
    "lr"
    "rf"
    "xgb"
    "lr_mc"
    "lr_pmc"
    "rf_mc"
    "rf_pmc"
    "xgb_mc"
    "xgb_pmc"
)
ntrials=50
seeds=$(cat seeds.txt | head -n $ntrials)

# job_submission_file="submission_file"

alphas=(
0.01
0.05
0.1
)
gammas=(
0.01
0.05
0.10
)
n_binses=(
5
10
)
rhos=(
0.001
0.01
0.1
)
ohc="ohc"
#https://stackoverflow.com/questions/38774355/how-to-parallelize-for-loop-in-bash-limiting-number-of-processes#38775799
num_procs=100
num_jobs="\j"  # The prompt escape for number of jobs currently running

for s in ${seeds[@]} ; do
    for alpha in ${alphas[@]} ; do
        for gamma in ${gammas[@]} ; do
            for n_bins in ${n_binses[@]} ; do
                for rho in ${rhos[@]} ; do
                    for m in ${methods[@]} ; do
                        while (( ${num_jobs@P} >= num_procs )); do
                            wait -n
                        done
                        python evaluate_model.py \
                            -ohc $ohc \
                            -file $dataset \
                            -results_path $rdir \
                            -seed $s \
                            -alpha $alpha \
                            -gamma $gamma \
                            -n_bins $n_bins \
                            -rho $rho \
                            -ml $m \
                            > "$rdir/bch_ohc[$ohc]_seed[$s]_alpha[$alpha]_gamma[$gamma]_n_bins[$n_bins]_rho[$rho]_ml[$m].log" \
                            &
                        ((++count))
                    done
                done
            done
        done
    done
done
# echo "created file $job_submission_file with $count jobs."
echo "ran $count jobs."

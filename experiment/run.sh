#!/usr/bin/bash
# Run experiment. 
rdir="../results/results_23-01-31/"
ntrials=100
seeds=$(cat seeds.txt | head -n $ntrials)

mkdir -p $rdir

alphas=(
0.001
0.01
0.1
0.2
)
gammas=(
0.01
0.05
)
n_binses=(
# 5
10
)
rhos=(
0.01
# 0.05
#0.05
0.1
# 0.2
# 0.5
)
methods=(
"lr"
"lr_mc"
"lr_pmc"
"rf"
"rf_mc"
"rf_pmc"
)
# all:
    # "lr"
    # "lr_mc"
    # "lr_mc_cv"
    # "lr_pmc"
    # "lr_pmc_cv"
    # "rf"
    # "rf_mc"
    # "rf_mc_cv"
    # "rf_pmc"
    # "rf_pmc_cv"
    # "xgb"
    # "xgb_mc"
    # "xgb_pmc"
groups=(
"ethnicity,gender"
"ethnicity,gender,insurance"
)
ohcs=(
    "ohc"
    "label_encoding"
)
# Job parameters
# https://stackoverflow.com/questions/38774355/how-to-parallelize-for-loop-in-bash-limiting-number-of-processes#38775799
# cores
N=96
num_jobs="\j"  # The prompt escape for number of jobs currently running
count=0

for s in ${seeds[@]} ; do
    for alpha in ${alphas[@]} ; do
        for gamma in ${gammas[@]} ; do
            for n_bins in ${n_binses[@]} ; do
                for rho in ${rhos[@]} ; do
                    for grouping in ${groups[@]} ; do
                        for ohc in ${ohcs[@]} ; do 
                            for m in ${methods[@]} ; do 
                                while (( ${num_jobs@P} >= N )); do
                                    wait -n
                                done
                                ((++count))
                                {
                                    job_name="${m}_seed${s}_alpha${alpha}_gamma${gamma}_n_bins${n_bins}_rho${rho}_group${grouping}_ohc${ohc}" 
                                    job_file="${rdir}/${job_name}" 

                                    echo "job $count = ${job_name}..." 

                                    python evaluate_model.py \
                                        -file data/mimic4_admissions.csv \
                                        -ml $m \
                                        -seed $s \
                                        -alpha $alpha \
                                        -gamma $gamma \
                                        -n_bins $n_bins \
                                        -rho $rho \
                                        -results_path $rdir \
                                        -groups $grouping \
                                        -ohc $ohc \
                                        | tee -i ${job_file} >/dev/null

                                    echo "$count completed ${job_file}..." 
                                } &
                            done
                        done 
                    done
                done
            done
        done
    done
done
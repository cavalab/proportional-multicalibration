#!/usr/bin/bash
# Run experiment. 
rdir="../results_22-05-14"
mkdir -p $rdir
job_submission_file="submission_file"

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
0.01
0.05
0.1
0.2
)
methods=(
    "lr"
    "lr_mc"
    "lr_pmc"
    "xgb"
    "xgb_mc"
    "xgb_pmc"
)
# "lr_mc_cv"
# "lr_pmc_cv"
    # "xgb_pmc"
    # "xgb_mc_cv"
    # "xgb_pmc_cv"
# )
ntrials=30
seeds=$(cat seeds.txt | head -n $ntrials)
# Job parameters
# cores
N=1
q="epistasis_long"
# mem=16384
# mem=8192
mem=4096

timeout=3600
count=0

for s in ${seeds[@]} ; do
    for alpha in ${alphas[@]} ; do
        for gamma in ${gammas[@]} ; do
            for n_bins in ${n_binses[@]} ; do
                for rho in ${rhos[@]} ; do
                    for m in ${methods[@]} ; do
                        # ((i=i%N)); ((i++==0)) && wait
                        job_name="${m}_seed${s}_alpha${alpha}_gamma${gamma}_n_bins${n_bins}_rho${rho}" 
                        job_file="${rdir}/${job_name}" 

                        bsub -o "${job_file}.out" \
                             -n $N \
                             -J $job_name \
                             -q $q \
                             -R "span[hosts=1] rusage[mem=${mem}]" \
                             -W $timeout \
                             -M $mem \
                             python evaluate_model.py \
                                 -file data/mimic4_admissions.csv \
                                 -ml $m \
                                 -seed $s \
                                 -alpha $alpha \
                                 -gamma $gamma \
                                 -n_bins $n_bins \
                                 -rho $rho \
                                 -results_path $rdir 
                                 # >> $job_submission_file
                       # echo python evaluate_model.py \
                       #           -file data/mimic4_admissions.csv \
                       #           -ml $m \
                       #           -seed $s \
                       #           -alpha $alpha \
                       #           -gamma $gamma \
                       #           -n_bins $n_bins \
                       #           -rho $rho \
                       #           -results_path $rdir > "jobs/input.${count}"
                        ((++count))
                    done
                done
            done
        done
    done
done
# echo "created file $job_submission_file with $count jobs."
echo "submitted $count jobs."
# echo bsub -J "multical[1-${count}]" cat input.\$LSB_JOBINDEX | bash 
# bsub -o "${job_file}.out" \
#      -n $N \
#      -J "multical[1-10000]" \
#      -q $q \
#      -R "span[hosts=1] rusage[mem=${mem}]" \
#      -W $timeout \
#      -M $mem \
#      cat input.\$LSB_JOBINDEX | bash 

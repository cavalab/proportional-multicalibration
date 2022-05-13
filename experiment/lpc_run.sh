#!/usr/bin/bash
# Run experiment. 
rdir="../results_22-05-13r1"

alpha=0.01
gamma=0.05
rho=0.1
n_bins=10

methods=(
    "lr_cv"
    "lr_mc"
    "lr_pmc"
    "xgb"
    "xgb_cv"
    "xgb_mc"
)
# "lr_mc_cv"
# "lr_pmc_cv"
    # "xgb_pmc"
    # "xgb_mc_cv"
    # "xgb_pmc_cv"
# )
seeds=$(cat seeds.txt)
# Job parameters
# cores
N=1
q="epistasis_long"
# mem=16384
# mem=8192
mem=4096

timeout=3600

for s in ${seeds[@]} ; do
    for m in ${methods[@]} ; do
        ((i=i%N)); ((i++==0)) && wait
        echo "python evaluate_model.py data/mimic4_admissions.csv -ml $m -seed $s -alpha $alpha -gamma $gamma -n_bins $n_bins -rho $rho -results_path $rdir"
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
    done
done


#!/usr/bin/bash
# Run experiment. 
rdir="results_22-05-13"

alpha=0.05
gamma=0.05
rho=0.05
n_bins=10

methods=(
    "xgb"
    "xgb_cv"
    "xgb_mc"
    "xgb_pmc"
    "xgb_mc_cv"
    "xgb_pmc_cv"
)
echo "methods:"
seeds=(
23654
15795
860
5390
16850
29910
4426
21962
14423
28020
)
echo "seeds:"
    echo $m
for m in ${methods[@]} ; do
    echo $m
    for s in ${seeds[@]} ; do
        echo $s
        python evaluate_model.py data/mimic4_admissions.csv \
            -ml $m \
            -seed $s \
            -alpha $alpha \
            -gamma $gamma \
            -n_bins $n_bins \
            -rho $rho \
            -results_path $rdir \
            &
    done
done


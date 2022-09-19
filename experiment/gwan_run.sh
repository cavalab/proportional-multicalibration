#!/usr/bin/bash
# Run chiefcomplaint experiment. 
rdir_bch="results/BCH/"
dataset="data/bch_final.csv"
# rdir_mimic="/home/guangya/multi-differential-calibration/results/MIMIC"
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
ohc=(
'1'
'0'
'-1'
)

ntrials=100
seeds=$(cat seeds.txt | head -n $ntrials)

for s in ${seeds[@]} ; do
    for m in ${methods[@]} ; do
        for text_coding in ${ohc[@]} ; do
            python evaluate_model.py \
                -file $dataset  \
                -seed $s \
                -ml $m \
                -ohc $text_coding \
                -results_path $rdir_bch
        done
    done
done

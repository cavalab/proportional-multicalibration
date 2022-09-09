#!/bin/sh
python evaluate_model.py -file preprocessing/final.csv -ml xgb -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/final.csv -ml xgb_mc -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/final.csv -ml xgb_pmc -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/final.csv -ml rf -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/final.csv -ml rf_mc -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/final.csv -ml rf_pmc -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/final.csv -ml lr -ohc -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/final.csv -ml lr_mc -ohc -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/final.csv -ml lr_pmc -ohc -results_path ../results/MIMIC/
python evaluate_model.py -file preprocessing/bch_final.csv -ml xgb -results_path ../results/BCH/
python evaluate_model.py -file preprocessing/bch_final.csv -ml xgb_mc -results_path ../results/BCH/
python evaluate_model.py -file preprocessing/bch_final.csv -ml xgb_pmc -results_path ../results/BCH/
python evaluate_model.py -file preprocessing/bch_final.csv -ml rf -results_path ../results/BCH/
python evaluate_model.py -file preprocessing/bch_final.csv -ml rf_mc -results_path ../results/BCH/
python evaluate_model.py -file preprocessing/bch_final.csv -ml rf_pmc -results_path ../results/BCH/
python evaluate_model.py -file preprocessing/bch_final.csv -ml lr -ohc -results_path ../results/BCH/
python evaluate_model.py -file preprocessing/bch_final.csv -ml lr_mc -ohc -results_path ../results/BCH/
python evaluate_model.py -file preprocessing/bch_final.csv -ml lr_pmc -ohc -results_path ../results/BCH/
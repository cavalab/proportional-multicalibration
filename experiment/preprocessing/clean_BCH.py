#https://github.com/cavalab/popp_fairness/issues/6
from distutils.command.clean import clean
from xmlrpc.client import boolean
from tqdm import tqdm
import ipdb
import os
from ast import arg
import numpy as np
import pandas as pd
import argparse
import importlib
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from collections import defaultdict
# demographics
A = ['Race','Ethnicity','Gender','Age']

A_rename = {
    'Hispanic Yes No': 'Ethnicity',
    'age_group': 'Age'
}

A_options = {
    'Race':defaultdict(lambda: 'Other'),
    'Ethnicity': {
       'Yes':'HL',
       'No':'NHL'
    },
    'Gender': {
        'M':'M',
        'F':'F'
    },
   'Age': {
       '>5Y':'older than 5Y',
       '18M-3Y':'5Y or younger', 
       '3-5Y':'5Y or younger', 
       '12-18M':'5Y or younger', 
       '0-3M':'5Y or younger', 
       '6-12M':'5Y or younger', 
       '3-6M':'5Y or younger'
   }
}

A_options['Race'].update({
        'Black or African American':'Black', 
        'White':'white', 
        'Asian':'Asian',
        'American Indian or Alaska Native':'AI',
        'Native Hawaiian or Other Pacific Islander':'NHPI', 
    })

# Need to use the data file, might need to merge dem with data file

def clean_dems(df):
    """Re-codes demographics according to dictionaries above"""
    df = df.rename(columns = A_rename)
    for a in A:
        if a in A_options.keys():
            df = df.loc[df[a].isin(list(A_options[a].keys())),:]
            df[a] = df[a].apply(lambda x: A_options[a][x])
    return df

def read_data(data):
    df = pd.read_csv(data)
    return df

COLUMNS_TO_DROP = ['Unnamed: 0', 'Contact Serial Number', 'ED Checkin Dt Tm',
       'ED Checkout Dt Tm', 'MRN','ED Derived Disposition','Race Line']

def rename_data(data):
    data.rename(columns={"Gender": "gender", "Ethnicity": "ethnicity"},inplace = True)

def process_data(data,dem,results_path = 'final.csv'):
    print('loading and processing BCH files...')
    df_dem = read_data(dem)

    df_dem = df_dem.drop(COLUMNS_TO_DROP,axis = 1)

    print('Renaming the categorical features.')
    df_dem = clean_dems(df_dem)

    df = df_dem
    rename_data(df)

    df = df.dropna(subset = ['isCase'])
    print('finished processing dataset.')
    print(f'size: {df.shape}, cases: {df.isCase.sum()/len(df)}')
    print('dataset columns:',df.columns)
    print(df.head())

    print('saving...')
    df.to_csv(results_path,index= False)
    print('done.')

# add more options here later
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Input the file location for BCH files", add_help=False)
    parser.add_argument('-bch_path', action='store', type=str,
                        default='/media/cavalab/data/popp/model.data.60mins_forBill/',
                        help='Path for admission file')
    parser.add_argument('-Data_File', action='store', type=str,
                        default='model.data.60mins_forBill.csv',
                        help='Path for Full Data')
    parser.add_argument('-Dem_File', action='store', type=str,
                        default='demographics_forBill.csv',
                        help='Path for Dem Data')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-p', action='store',
                        dest='PATH',default='bch_final.csv',type=str,
            help='Path of Saved final fire')
    args = parser.parse_args()

    process_data(os.path.join(args.bch_path, args.Data_File), 
                 os.path.join(args.bch_path, args.Dem_File), 
                 results_path = args.PATH) # Rerun this file again to include prev_adm
 
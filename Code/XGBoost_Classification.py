# !/usr/bin/env python3

## XGBoost Classification Models

import os
from pathlib import Path
import sys

### Basic Directories
# Current directory (code)
cwd = os.getcwd()
# Parent directory (root directory for all folders)
parent = Path(cwd).parent
# Modules directory
modules_dir = os.path.join(cwd,'modules')
# Dataset directory
dsets_dir = os.path.join(parent,'Dataset')
# Analysis Results Data Directory
anres_data_dir = os.path.join(parent,'Analysis_results')
if not os.path.exists(anres_data_dir):
    os.makedirs(anres_data_dir)
# Evaluation Results Directory
res_dir = os.path.join(parent,'Evaluation_Results')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# Directory to store collected RAW data
res_raw_dir = os.path.join(res_dir,'Collect_Analysis_results')
if not os.path.exists(res_raw_dir):
    os.makedirs(res_raw_dir)
# Directory to store results and plots for Statistical Analysis
res_stat_dir = os.path.join(res_dir, 'Statistical_tests')
if not os.path.exists(res_stat_dir):
    os.makedirs(res_stat_dir)

# Include modules' dir into system to import modules
sys.path.append(modules_dir)


### Imports from modules
from Preprocess_dataset import Preprocess_dataset
from Dataset_CLF_analysis import Dataset_CLF_analysis


def main():
    #######  PreProcess RAW data  #######
    ###  Link to dataset with RAW data
    # dset_n = 'RAW_data.xlsx'
    dset_n = 'RAW_data.xlsx'
    dset_lnk = os.path.join(dsets_dir, dset_n)

    ### Preprocess RAW data
    ppdset = Preprocess_dataset(dset_lnk)
    data_df = ppdset.preproc_dset


    #######  Analyse datasets with XGBoost Classification algorithm  #######
    dset = Dataset_CLF_analysis()
    ### Define dependent and independent variables
    y = data_df['Syntax Score']
    X = data_df.drop(['Syntax Score'], axis=1)
    ### Transform dependent variable's values to create 2 groups
    # Healthy (SS=0->0) vs Patients (SS>0->1)
    y[y>0] = 1
    ### Set analysis parameters
    # Metric to use
    metric_n = 'logloss'
    # Number of predictions cv iterations
    pred_splits = 10
    # Number of optimization cv iterations
    opt_cv = 10
    # Number of random parameters sets to examine during optimization
    opt_iters = 100
    # Number of analysis iterations
    n = 10
    # Probas threshold for groups
    thresh = 0.5
    # Methods to be used in analysis
    use_models = ['XGBClf']
    ### Perform analysis of dataset
    dset.analyse_datasets(X, y, anres_data_dir, use_models=use_models, metric_n=metric_n, 
                        pred_splits=pred_splits, opt_cv=opt_cv, opt_iters=opt_iters, n=n,
                        thresh=thresh)
    

    #######  Collect RAW Experimental Data  #######
    # Method to collect all analysis results
    dset.collect_raw_res_data(anres_data_dir)
    # Export results (pickle and excel)
    rd_dic_fn = 'Collected_experimental_data.xlsx'
    rd_dic_fn_lnk = os.path.join(res_raw_dir, rd_dic_fn)
    dset.export_dict_to_file(dset.raw_data_dic, rd_dic_fn_lnk, kl=1, unif_col_widths=1)
    # Export results to pickle (The whole object is stored)
    rd_dic_fn = 'Collected_experimental_data.pickle'
    rd_dic_fn_lnk = os.path.join(res_raw_dir, rd_dic_fn)
    dset.export_dict_to_file(dset, rd_dic_fn_lnk, kl=1)


    #######  Experimental Data Analysis  #######
    dset.analyze_experimental_data(res_raw_dir, res_stat_dir)



if __name__ == '__main__':
    main()
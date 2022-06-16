#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold


    
def set_inner_gridsearch_cv(estimator_ppln, model_params, scoring="neg_mean_absolute_error", n_iters=100,
                            n_splits=10, random_state=None):
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid_srch = RandomizedSearchCV(estimator=estimator_ppln,
                                   param_distributions=model_params,
                                   scoring=scoring,
                                   n_iter=n_iters,
                                   n_jobs=-1,
                                   cv=inner_cv,
                                   verbose=0,
                                   refit=True)
    return(grid_srch)


def save_res_excel(full_path, preds_df, metr_df):
    if(os.path.isfile(full_path)):
        base_name = ('.').join(full_path.split('.')[:-1])
        extens = full_path.split('.')[-1]
        i = 1
        while(os.path.isfile(full_path)):
            full_path = '%s_%d.%s' %(base_name, i, extens)
            i +=  1
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')   
    workbook=writer.book
    # Predictions
    sht1 = 'Predictions'
    worksheet=workbook.add_worksheet(sht1)
    writer.sheets[sht1] = worksheet
    preds_df.to_excel(writer, sheet_name=sht1, startrow=0 , startcol=0, merge_cells=True)
    # Metrics
    sht2 = 'Metrics'
    worksheet=workbook.add_worksheet(sht2)
    writer.sheets[sht2] = worksheet
    metr_df.to_excel(writer, sheet_name=sht2, startrow=0 , startcol=0, merge_cells=True)
    writer.save()
    

### Exports a data dictionary to a multisheet excel file
def export_dict_to_file(data_dic, store_link, keep_index=True, kl=1, unif_col_widths=False):
    if(store_link.endswith('.pickle')):
        with open(store_link, 'wb') as handle:
            pickle.dump(data_dic, handle)
    elif(store_link.endswith('.xlsx')):
        writer = pd.ExcelWriter(store_link, engine='xlsxwriter')
        workbook=writer.book
        if(kl==1):
            keys1 = data_dic.keys()
            for key1 in keys1:
                df = data_dic[key1]
                sht = '%s' %(key1)
                df.to_excel(writer, sheet_name=sht, index=keep_index, startrow=0 , startcol=0, merge_cells=True)
                worksheet=workbook.get_worksheet_by_name(sht)
                df_widths = dataframe_columns_widths(df, incl_index=keep_index, unif_col_widths=unif_col_widths)
                for i, width in enumerate(df_widths):
                    width = 3 if width<3 else width
                    worksheet.set_column(i, i, width*1.1)
        writer.save()


def dataframe_columns_widths(df, incl_index=True, unif_col_widths=False):
    df_widths = list()
    if(incl_index):
        idx_widths = list()
        for idx in df.index:
            idx_widths.append(len(str(idx)))
        df_widths.append(np.max(idx_widths))
    for col in df.columns:
        widths = list()
        widths.append(len(str(col)))
        for val in df[col].values:
            widths.append(len(str(val)))
        df_widths.append(np.max(widths))
    if(unif_col_widths):
        ucw = np.max(df_widths[1:])
        if(unif_col_widths>1):
            ucw = unif_col_widths
        if(incl_index):
            df_widths = df_widths[:1] + [ucw]*len(df_widths[1:])
        else:
            df_widths = [ucw]*len(df_widths)
    
    return(df_widths)    









#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Class Preprocess_datasets  ####

import pandas as pd


class Preprocess_dataset():
    
    preproc_dset = None
    
    def __init__(self, dset_lnk):
        dset = pd.read_excel(dset_lnk, index_col=0)
        cols = dset.columns
        dset.columns = [col.replace('<', ' lt ') if '<' in col else col for col in cols]
        pp_dset = pd.get_dummies(dset, drop_first=True)
        new_cols = [col for col in pp_dset.columns if col != 'Syntax Score'] + ['Syntax Score']
        self.preproc_dset = pp_dset[new_cols]
        out_lnk = dset_lnk.replace('.xlsx', '_preproc.xlsx')
        self.export_dataframe_to_file(self.preproc_dset, out_lnk, keep_index=True)


    ### Exports preprocesseed dataset to an excel file
    def export_dataframe_to_file(self, df, store_link, keep_index=True):
        writer = pd.ExcelWriter(store_link, engine='xlsxwriter')
        workbook=writer.book
        sht = 'PreProc_data'
        df.to_excel(writer, sheet_name=sht, index=keep_index, startrow=0 , startcol=0, merge_cells=True)
        worksheet=workbook.get_worksheet_by_name(sht)
        worksheet.set_column(0, 1, 12)
        writer.save()




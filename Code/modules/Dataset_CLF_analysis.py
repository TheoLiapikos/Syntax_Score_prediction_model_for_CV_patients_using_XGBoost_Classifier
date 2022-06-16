#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Class Dataset_ML_analysis  ####
# Collects all functions for dataset analysis by selected ML methods

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import io
import os
from datetime import datetime
from sys import stdout

from scipy.stats import randint
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, log_loss

from Analysis_methods import set_inner_gridsearch_cv, save_res_excel, export_dict_to_file #, models_shortnames_dic,\
from Evaluation_methods import compar_boxplots, roc_auc_multiplots, thresh_tune


metrics_dic = {
    'logloss': 'neg_log_loss',
    }

metrics = ['Accuracy', 'Log_Loss', 'ROC_AUC', 'F1', 'Recall', 'Precision']

models_dic = {
    'XGBClf': XGBClassifier,
    }

models_shortnames_dic = {
    'XGBClf': 'XGBC',
}

models_params_dic = {
    'XGBClf': {'n_estimators': randint(10,200), 'max_depth': randint(1,12), 'learning_rate': uniform(0.01,0.25),
               'gamma': uniform(0.0,10.0), 'reg_alpha': uniform(0.0,10.0), 'reg_lambda': uniform(0.0,10.0), 'objective': ['reg:squarederror'],},
    }




class Dataset_CLF_analysis():
    
    raw_data_dic = dict()
    statistical_data_dic = dict()
    comparative_boxplots_dic = dict()
    roc_auc_plots_dic = dict()
    thresholds_tuning_dic = dict()
    
    def __init__(self):
        pass
    
    
    def mape(self, actual, pred): 
        actual, pred = np.array(actual), np.array(pred)
        return (np.mean(np.abs((actual - pred) / actual)) * 100)
    
    
    def analyse_datasets(self, X, y, dset_store_lnk, use_models=list(), metric_n='ACCU',
                         pred_splits=10, opt_cv=10, opt_iters=200, n=1, thresh=0.5):
        metric = metrics_dic[metric_n]
        for i in range(1, n+1):
            print('\nRound %d of %d' %(i, n))
            # Computed metrics values
            met_ls = list()
            # predictions list
            pred_ls = list()
            # Model used for predictions
            for model_n in use_models:   ### TO_DO: REMOVE FOR
                print('\nStarting analysis with method: %s (%s)' %(model_n, datetime.now().strftime("%H:%M")))
                model = models_dic[model_n]()
                params = models_params_dic[model_n]
                ### Get models predictions
                ypr = self.analyse_dataset(X, y, model, params, metric=metric, pred_splits=pred_splits,
                                           opt_cv=opt_cv, opt_iters=opt_iters, rs=None)
                ypr.name = model_n
                pred_ls.append(ypr)
                ### Calculate Metrics values
                # For ROC-AUC use directly probas
                # For rest metrics turn probas into class prediction (using specific threshold to belong to class 1)
                yp_class = ypr > thresh
                metr = {
                    "Accuracy": accuracy_score(y, yp_class),
                    "MAPE": self.mape(y, yp_class),
                    "ROC_AUC": roc_auc_score(y, ypr, multi_class='ovo'),
                    "F1": f1_score(y, yp_class, average='weighted'),
                    "Recall": recall_score(y, yp_class, average='weighted'),
                    "Precision": precision_score(y, yp_class, average='weighted'),
                }
                metr = pd.Series(metr)
                metr.name = model_n
                met_ls.append(metr)
            print('\n%s' %datetime.now().strftime("%H:%M"))
            # Concat all predictions
            y.name='Syntax_Score'
            preds_df = pd.concat([y, np.round(pd.concat(pred_ls, axis=1),2)], axis=1)
            # Concat all computed metrics
            metr_df = pd.concat(met_ls, axis=1)
            print(metr_df)
            # Create an excel file and save current round results
            exl_n = 'CLF_(%s_%d_%d)_preds_metrics.xlsx' %(metric_n, pred_splits, opt_cv)
            # Full path to excel file
            full_path = os.path.join(dset_store_lnk, exl_n)
            save_res_excel(full_path, preds_df, metr_df)

    
    def analyse_dataset(self, X, y, model, params, metric='accuracy', pred_splits=10, opt_cv=10, opt_iters=100,
                    rs=42, print_iters=True):
        # Create Estimator pipeline
        est_ppln = Pipeline([
            ('std', StandardScaler()),
            ('clf', model)
        ])
        # Modify hyperparameters naming to use in optimization gridsearch procedure
        params = {'%s__%s'%('clf', i):j for i,j in params.items()}
        # Create hyperparameter optimization gridsearch (inner CV)
        grid_srch = set_inner_gridsearch_cv(est_ppln, params, scoring=metric, n_iters=opt_iters, n_splits=opt_cv, random_state=rs)
        # Create and apply outer CV procedure
        outer_cv = StratifiedKFold(n_splits=pred_splits, shuffle=True, random_state=rs)   #### STRATIFIED
        # Collect predictions from all iterations
        y_preds = list()
        for iter_n, (tr_index, te_index) in enumerate(outer_cv.split(X, y)):
            if(print_iters):
                stdout.write('\rOuter CV iteration %d of %d' %(iter_n+1, pred_splits))
                stdout.flush()
            X_tr = X.iloc[tr_index]
            X_te = X.iloc[te_index]
            y_tr = y.iloc[tr_index]
            y_te = y.iloc[te_index]
            ### Scale data
            # Train scaler on X_train and apply on X_test
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            grid_srch.fit(X_tr, y_tr)
            ### Predictions using Probas
            preds = grid_srch.predict_proba(X_te)
            # Keep only proba for 1 class
            preds = np.array([pred[1] for pred in preds])
            preds = pd.Series(preds.flatten(), index=y_te.index)
            y_preds.append(preds)
        # Concat and return predictions
        y_preds = pd.concat(y_preds).loc[y.index]
        return(y_preds)
    

    def collect_raw_res_data(self, raw_data_lnk):
        tot_res_dic = dict()
        tot_fns = os.listdir(raw_data_lnk)

        inter_metrs = list()
        inter_preds = list()
        # Get all EXCEL files present in that subfolder
        tot_pred_dfs = [pd.read_excel(os.path.join(raw_data_lnk, fn), sheet_name='Predictions', index_col=0) for fn in tot_fns if fn.endswith('.xlsx')]
        tot_metr_dfs = [pd.read_excel(os.path.join(raw_data_lnk, fn), sheet_name='Metrics', index_col=0) for fn in tot_fns if fn.endswith('.xlsx')]
        # For each couple of Predictions-Metrics
        for i, pred_df in enumerate(tot_pred_dfs):
            metr_df = tot_metr_dfs[i]
            for method in metr_df.columns:
                if(method not in models_shortnames_dic):
                    continue
                metrs = metr_df.loc[:,method]
                metrs.index = [idx.split(':')[0] for idx in metrs.index] # <----
                preds = pred_df.loc[:,method]
                
                main_sub = models_shortnames_dic[method]
                main_sub_ser = pd.Series([main_sub, i+1], index=['Method', 'Iter'])
                metrs_f = pd.concat([main_sub_ser, metrs])
                preds_f = pd.concat([main_sub_ser, preds])
                inter_metrs.append(metrs_f)
                inter_preds.append(preds_f)

        metrs_df_f = pd.concat(inter_metrs, axis=1).T
        metrs_df_f = metrs_df_f.set_index(['Method', 'Iter'])
        metrs_df_f.sort_values(['Method', 'Iter'], inplace=True)
        tot_res_dic['Metrics'] = metrs_df_f

        preds_df_f = pd.concat(inter_preds, axis=1).T
        preds_df_f = preds_df_f.set_index(['Method', 'Iter'])
        preds_df_f.sort_values(['Method', 'Iter'], inplace=True)
        # Incorporate Experimental SS values at first line
        exp_rt_head = pd.Series(['Exp_Syntax_Score', 0], index=['Method', 'Iter'])
        exp_rt_df = pd.DataFrame(pd.concat([exp_rt_head, tot_pred_dfs[0]['Syntax_Score']])).T.set_index(['Method', 'Iter']).astype(int)
        preds_df_f = pd.concat([exp_rt_df, preds_df_f])
        tot_res_dic['Predictions'] = preds_df_f

        ### Procedure to calculate LogLoss metric in place of MAPE
        y_true = tot_res_dic['Predictions'].iloc[0,:]
        lloss_lst = list()### Compute Confidence Intervals for ROC-AUC series - Bootstraping approach
        for i in range(1,len(tot_res_dic['Predictions'].index)):
            y_pred = tot_res_dic['Predictions'].iloc[i,:]
            lloss_lst.append(log_loss(y_true.tolist(),y_pred.tolist()))
        tot_res_dic['Metrics'].rename({'MAPE':'Log_Loss'}, inplace=True, axis=1)
        tot_res_dic['Metrics']['Log_Loss'] = lloss_lst

        self.raw_data_dic = tot_res_dic


    def export_dict_to_file(self, data_dic, store_link, keep_index=True, kl=1, unif_col_widths=0):
        export_dict_to_file(data_dic, store_link, keep_index=keep_index, kl=kl, unif_col_widths=unif_col_widths)
    
    
    def analyze_experimental_data(self,res_dir, stat_dir):
        metrics_ls = ['Accuracy', 'Log_Loss', 'ROC_AUC', 'F1', 'Recall', 'Precision']
        # Pickles files (datasets) in collected RAW data subdir
        raw_dsets_pckls = [f for f in os.listdir(res_dir) if os.path.isfile(os.path.join(res_dir, f)) and f.endswith('.pickle')]
        for raw_dsets_pckl in raw_dsets_pckls:
            self.load_raw_data(os.path.join(res_dir, raw_dsets_pckl))
            self.compute_statistical_data()
            store_link = os.path.join(stat_dir, 'Collected_metrics_values.xlsx')
            self.export_dict_to_file(self.statistical_data_dic, store_link, keep_index=False, kl=1)
            # Create and store comparative boxplots for all metrics
            self.comparative_metric_boxplots(metrics=metrics_ls, col_width=6, dropna=False,
                                            fillna='median', show_plot=False)
            bxp_lnk = os.path.join(stat_dir, 'Models_performance_comparative_BoxPlots.xlsx')
            self.export_plots_dict_to_file(plots_dic=self.comparative_boxplots_dic, store_link=bxp_lnk, kl=1)
            # Create and store ROC AUC plots (with Confidence Intervals)
            for alpha in [0.95, 0.99]:
                self.roc_auc_plots(cols_n=4, cols_width=4, alpha=alpha, show_plot=False)
                ci = alpha*100
                rap_lnk = os.path.join(stat_dir, 'ROC_AUC_plots_(CI_%.1f%%).xlsx' %ci)
                self.export_plots_dict_to_file(plots_dic=self.roc_auc_plots_dic, store_link=rap_lnk, kl=1)
            # Fine tune thresholds for classes probabilities
            for alpha_thr in [0.05, 0.01]:
                self.thresholds_tuning(alpha_thr)
                tt_lnk = os.path.join(stat_dir, 'Thresholds_tuning_(alpha=%s).xlsx' %str(alpha_thr))
                self.export_dict_to_file(self.thresholds_tuning_dic, tt_lnk, keep_index=True, kl=1)


    def load_raw_data(self, raw_data_obj_lnk):
        if(raw_data_obj_lnk.endswith('.pickle')):
            with open(raw_data_obj_lnk, 'rb') as handle:
                rd_obj = pickle.load(handle)
                self.raw_data_dic = rd_obj.raw_data_dic
    
    
    def compute_statistical_data(self):
        models = list(models_shortnames_dic.values())
        metrics = self.raw_data_dic['Metrics'].columns.to_list()
        st_data_dic = {i:'' for i in metrics}
        for metric in metrics:
            metr_res_lst = list()
            for method in models:
                if method not in self.raw_data_dic['Metrics'].index.levels[0]:
                    continue
                metr_mod_res = self.raw_data_dic['Metrics'].loc[(method,), metric]
                metr_mod_res.name=method
                metr_mod_res.reset_index(drop=True, inplace=True)
                metr_res_lst.append(metr_mod_res)
            st_data_dic[metric] = pd.concat(metr_res_lst, axis=1)
        self.statistical_data_dic = st_data_dic    


    def comparative_metric_boxplots(self, metrics=list(), col_width=8, dropna=False, fillna='median',
                                    show_plot=False):
        if(len(self.statistical_data_dic)==0):
            self.compute_statistical_data()
        if(len(metrics)==0):
            print('Define list of metrics to use...')
        else:
            self.comparative_boxplots_dic = compar_boxplots(self.statistical_data_dic, metrics=metrics,
                                                            col_width=col_width, dropna=dropna, fillna=fillna,
                                                            show_plot=show_plot)    


    def export_plots_dict_to_file(self, plots_dic, store_link, kl=1):
        if(store_link.endswith('.pickle')):
            with open(store_link, 'wb') as handle:
                pickle.dump(plots_dic, handle)
        elif(store_link.endswith('.xlsx')):
            writer = pd.ExcelWriter(store_link, engine='xlsxwriter')
            workbook=writer.book
            if(kl==1):
                keys1 = plots_dic.keys()
                for key1 in keys1:
                    # Plot figure to store
                    fig = plots_dic[key1]
                    sht_n = '%s' %(key1)
                    sht = workbook.add_worksheet(sht_n)
                    imgdata=io.BytesIO()
                    fig.savefig(imgdata, format='png')
                    sht.insert_image(1, 1, '', {'image_data': imgdata})
                    sht.hide_gridlines(2)
            if(kl==2):
                keys1 = list(plots_dic.keys())
                keys2 = list(plots_dic[keys1[0]].keys())
                for key1 in keys1:
                    for key2 in keys2:
                        fig = plots_dic[key1][key2]
                        sht_n = '%s_%s' %(key1, key2)
                        sht = workbook.add_worksheet(sht_n)
                        imgdata=io.BytesIO()
                        fig.savefig(imgdata, format='png')
                        sht.insert_image(1, 1, '', {'image_data': imgdata})
                        sht.hide_gridlines(2)
            writer.save()    


    def roc_auc_plots(self, cols_n=4, cols_width=4, alpha=0.95, show_plot=False):
        preds_df = self.raw_data_dic['Predictions'].copy()
        y_exp = preds_df.loc[('Exp_Syntax_Score',0),:]
        y_exp.name = 'y_exp'
        preds_df.drop(('Exp_Syntax_Score',0), inplace=True)
        methods = [meth for meth in models_shortnames_dic.values() if meth in preds_df.index.levels[0]]
        y_preds = list()
        for method in methods:
            # Get the mean probas of all iterations
            ypred = preds_df.loc[(method,),:].mean()
            ypred.name=method
            y_preds.append(ypred)
        self.roc_auc_plots_dic = roc_auc_multiplots(y_exp, y_preds, cols_n, cols_width, alpha=alpha,
                                                    show_plot=show_plot)
    
    
    def thresholds_tuning(self, alpha=0.05):
        preds_df = self.raw_data_dic['Predictions'].copy()
        y_exp = preds_df.loc[('Exp_Syntax_Score',0),:]
        y_exp.name = 'y_exp'
        preds_df.drop(('Exp_Syntax_Score',0), inplace=True)
        methods = [meth for meth in models_shortnames_dic.values() if meth in preds_df.index.levels[0]]
        y_preds = list()
        for method in methods:
            # Get the mean probas of all iterations
            ypred = preds_df.loc[(method,),:].mean()
            ypred.name=method
            y_preds.append(ypred)
        self.thresholds_tuning_dic = thresh_tune(y_exp, y_preds, alpha=alpha)





    

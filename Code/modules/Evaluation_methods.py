#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Basic functions to compute analytical rankings for all methods and for all datasets  ####

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score,\
                            fbeta_score, recall_score, precision_score, log_loss

import matplotlib.pyplot as plt



def compar_boxplots(st_data_dic, metrics=list(), col_width=8, dropna=False, fillna='median', show_plot=False):
    boxplots_dic = dict()
    metrics_to_plot = st_data_dic.keys() if len(metrics)==0 else metrics
    cols_n = 1
    rows_n = len(metrics_to_plot)
    row_height = col_width/2
    fig, axs = plt.subplots(rows_n, cols_n, figsize=(cols_n*col_width, rows_n*row_height))
    for i, metric in enumerate(metrics_to_plot):
        try:
            df = st_data_dic[metric]
        except:
            continue
        if(dropna):
            df = df.dropna().reset_index(drop=True)
        else:
            df = df.fillna(df.median()) if(fillna=='median') else df.fillna(df.mean())
        ax = axs[i]
        df_lst = [df[col] for col in df.columns]
        ax.boxplot(df_lst, showmeans=True)
        ax.set_ylabel(metric, fontsize=col_width+4, weight='bold')
        ax.set_xticklabels(df.columns)
        # ax.set_title(method, fontsize=15, weight='bold')
        ax.tick_params(axis='both', labelsize=col_width+2)
        plt.rcParams['figure.dpi']=300
        plt.tight_layout()
        if(not show_plot):
            plt.close()
    boxplots_dic['Comparative_Boxplots'] = fig
    return(boxplots_dic)


def roc_auc_multiplots(y_exp, y_preds, cols_n, cols_width, alpha=0.95, show_plot=False):
    roc_auc_plots_dic = dict()
    rows_n = int(np.ceil(len(y_preds)/cols_n))
    # rows_height = 2*cols_width/3
    rows_height = cols_width

    fig, axs = plt.subplots(rows_n, cols_n, figsize=(cols_n*cols_width, rows_n*rows_height))

    for i, y_pred in enumerate(y_preds):
        y_pred = y_preds[i]
        fpr, tpr, threshold = roc_curve(y_exp.to_list(), y_pred)
        roc_scr = roc_auc_score(y_exp.to_list(), y_pred, multi_class='ovo')
        # Confidence Intervals
        ci_lo, ci_up = roc_auc_ci_bootstrap(y_exp, y_pred, alpha=alpha)
        ax = axs.flatten()[i]
        ax.plot(fpr, tpr, 'b', label='AUC = %0.3f (%0.3f-%0.3f)' %(roc_scr, ci_lo, ci_up))
        ax.plot([0, 1], [0, 1],'r--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(y_pred.name)
        ax.set_ylabel('True Positive Rate') # or 'Sensitivity'
        ax.set_xlabel('False Positive Rate') # or '1-Specificity'
        ax.legend(loc='lower right')
    plt.rcParams['figure.dpi']=300
    plt.tight_layout()
    for j in np.arange(i+1, rows_n*cols_n):
        ax = axs.flatten()[j]
        ax.set_axis_off()
    if(not show_plot):
        plt.close()
    roc_auc_plots_dic['ROC_AUC_plots'] = fig
    return(roc_auc_plots_dic)


def thresh_tune(y_exp, y_preds, alpha=0.05):
    res_dic = dict()
    thresholds = np.linspace(0, 1, 101)
    for i, y_pred in enumerate(y_preds):
        fn_lst = list()
        tp_lst = list()
        # Number of true positive samples
        pos = y_exp.sum()
        for thresh in thresholds:
            y_pred_class = y_pred > thresh
            tn, fp, fn, tp = confusion_matrix(list(y_exp), y_pred_class).ravel()
            fn_lst.append(fn)
            tp_lst.append(tp)
        ### Working with True Positive disribution
        tp_proportion = pd.Series(tp_lst, index=thresholds)/pos
        thresh_man = tp_proportion[tp_proportion<=(1-alpha)].index[0]
        metr_res = list()
        for thresh in [0.5, thresh_man]:
            yp_class = y_pred > thresh
            tn, fp, fn, tp = confusion_matrix(list(y_exp), yp_class).ravel()
            accu = accuracy_score(list(y_exp), yp_class)
            logloss = log_loss(y_exp.tolist(),y_pred.tolist())
            rocauc =  roc_auc_score(list(y_exp), y_pred, multi_class='ovo')
            f1 = f1_score(list(y_exp), yp_class, average='weighted')
            f2 = fbeta_score(list(y_exp), yp_class, beta=2, average='weighted')
            rec = recall_score(list(y_exp), yp_class, average='weighted')
            prec = precision_score(list(y_exp), yp_class, average='weighted')
            res = pd.Series(
                [thresh, accu, logloss, rocauc, f1, f2, rec, prec, tn, fn, tp, fp],
                index = ['Threshold', 'Accuracy', 'Log_Loss', 'ROC_AUC', 'F1', 'F2', 'Recall', 'Precision',
                        'True Negative', 'False Negative', 'True Positive', 'False Positive'],
                name = y_pred.name
            )
            metr_res.append(res)
        res_dic['Thresholds_tuning'] = np.round(pd.concat(metr_res, axis=1), 3)
    return(res_dic)


def roc_auc_ci_bootstrap(y_true, y_pred, alpha=0.95, n_bootstraps=1500):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # Bootstrap
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(list(y_true[indices]), list(y_pred[indices]))#, multi_class='ovo')
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    # Compute %CI bounds
    confidence_lower = sorted_scores[int((0.5-alpha/2) * len(sorted_scores))]
    confidence_upper = sorted_scores[int((0.5+alpha/2) * len(sorted_scores))]
    return(confidence_lower, confidence_upper)






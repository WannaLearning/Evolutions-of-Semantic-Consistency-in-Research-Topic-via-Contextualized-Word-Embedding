#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 20:02:33 2023

@author: aixuexi
"""
import re
import os
import sys
import math
import json
import time as time_lib
import pickle
import random
import multiprocessing
import pandas as pd
import numpy as np
import prettytable as pt
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from tqdm import tqdm
from sklearn import decomposition
from matplotlib import rcParams
from adjustText import adjust_text 

from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix, csr_matrix

import torch
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer

import tslearn
from tslearn.metrics import dtw, dtw_path
from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.datasets import grunfeld
from linearmodels.panel import PanelOLS
from linearmodels.panel import PooledOLS
from linearmodels.datasets import wage_panel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from Utils import save_file, read_file, abs_file_path
from CalculateMetrics import get_vec_func, calculate_self_similarity_func, maximum_explainable_variance_func, calculate_self_distance_func



#%%
"""  
    Section 5.1
    Checking the quality of topic embeedings 
    Show and exclude Anisotropy in our metrics
"""
FoS2Vec_path = os.path.join(abs_file_path, "FoS2Vec")
FoSs = os.listdir(FoS2Vec_path)


def sampling_control_group(beg_year, end_year):
    """ random sample embeddings from different topics to cnostruct vectors pool """
    FoS2Vec_path = os.path.join(abs_file_path, "FoS2Vec")
    FoSs = os.listdir(FoS2Vec_path)
    sampling_size_2 = 100
    vecs_2 = list()
    for fid in tqdm(FoSs):
        dic  = read_file(os.path.join(FoS2Vec_path, fid))
        vecs = get_vec_func(dic, beg_year, end_year)
        if len(vecs) == 0:
            continue
        if len(vecs) <= sampling_size_2:
            vecs_2.append(vecs)
        else:
            selected_index = np.arange(len(vecs))
            random.shuffle(selected_index)
            selected_index = selected_index[:sampling_size_2]
            vecs_2.append(vecs[selected_index, :])
    vecs_2 = np.concatenate(vecs_2, axis=0)
    shuffle_index = np.arange(len(vecs_2))
    random.shuffle(shuffle_index)
    vecs_2 = vecs_2[shuffle_index, :]
    save_file(vecs_2, "./temp/vecs_2.pkl")


def check_for_anisotropy_from_self_similarity(sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio):
    """ demostrate the anisotropy in the topic embeddings """
    # 说明 相同FoS的向量相似性 > 不同FoS的向量相似性

    path_same = "./temp/self_sim_same_dis.pkl"
    path_diff = "./temp/self_sim_diff_dis.pkl"
    path_rand = "./temp/vecs_2.pkl" # 随机向量池
    
    if os.path.exists(path_same) and os.path.exists(path_diff):
        self_sim_same_dis = read_file(path_same)
        self_sim_diff_dis = read_file(path_diff)
    else:
        if os.path.exists(path_rand):
            vecs_2 = read_file(path_rand)
        else:
            vecs_2 = sampling_control_group(1990, 2018)
        
        thresh = 100
        self_sim_same_dis = dict()
        self_sim_diff_dis = dict()
        for fid in tqdm(FoSs):
            # fid  = 'artificial neural network.pkl'
            dic  = read_file(os.path.join(FoS2Vec_path, fid))
            vecs = get_vec_func(dic, 1990, 2018)
            if len(vecs) >= thresh:    
                self_sim_same_dis[fid] = calculate_self_similarity_func(vecs, vecs,   True,  sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
                self_sim_diff_dis[fid] = calculate_self_similarity_func(vecs, vecs_2, False, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
        save_file(self_sim_same_dis, path_same)
        save_file(self_sim_diff_dis, path_diff)

    # 绘制分布
    x = list(self_sim_diff_dis.keys())
    y_diff = np.array([self_sim_diff_dis[i] for i in x])
    y_same = np.array([self_sim_same_dis[i] for i in x])
    y_sub  = y_same - y_diff
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    ax1 = sns.distplot(y_same, hist=True, kde=True, rug=False,
                   bins=15,
                   # fit=norm,
                   hist_kws = {'rwidth':1, 'color':'lime', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.75, "label": r"$SSIM(V_j(:t))$"},
                   kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": ""},
                   # rug_kws  = {"color": "black",   "alpha":0.25, "linewidth": 0.01, "height":0.05},
                   # fit_kws  = {"color": "black", "alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": "Normal"},
                   )
    sns.distplot(y_sub, hist=True, kde=True, rug=False,
                 bins=15,
                 # fit=norm,
                 hist_kws = {'rwidth':1, 'color':'blue', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.5, "label": r"$SSIM(V_j(:t))$ - $SSIM(V_j(:t), \widehat{V_j(:t)})$"},
                 kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": "KDE"},
                 # rug_kws  = {"color": "black",  "alpha":0.25, "linewidth": 0.01, "height":0.05},                  # fit_kws  = {"color": "black", "alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": ""}
                 ax = ax1)
    ax2 = plt.twinx()
    ax2 = sns.distplot(y_diff, hist=True, kde=True, rug=False,
                       bins=15,
                       # fit=norm,
                       hist_kws = {'rwidth':1, 'color':'gray', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.75, "label": "$SSIM(V_j(:t), \widehat{V_j(:t)})$"},
                       kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": ""},
                       # rug_kws  = {"color": "black", "alpha":0.25, "linewidth": 0.01, "height":0.05},
                       # fit_kws  = {"color": "black","alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": ""},
                       ax=ax2)

    
    ax1.set_xlabel(r"Self similarity ($t$=2018)")
    ax1.set_ylabel("Density")
    ax2.set_ylabel("")
    ax1.legend(loc='upper left',  frameon=False, fontsize=15)
    ax2.legend(loc='upper right', frameon=False, fontsize=15)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_xlim(0, 1)
    ax1.set_yticks(np.arange(0, 35, 5))
    ax2.set_yticks(np.arange(0, 35, 5))
    ax2.tick_params(axis='y',colors='gray')


def check_for_anisotropy_from_maximum_explainable_variance():
    """ demostrate the anisotropy in the topic embeddings """
    # 说明相同FoS的向量的MER更大
    
    path_same = "./temp/mer_same_dis.pkl"
    path_diff = "./temp/mer_diff_dis.pkl"
    path_rand = "./temp/vecs_2.pkl" # 随机向量池
    
    if os.path.exists(path_same) and os.path.exists(path_diff):
        mer_same_dis = read_file(path_same)
        mer_diff_dis = read_file(path_diff)
    else:
        if os.path.exists(path_rand):
            vecs_2 = read_file(path_rand)
        else:
            vecs_2 = sampling_control_group(1990, 2018)
    
        thresh = 100
        mer_same_dis = dict() 
        mer_diff_dis = dict()
        for fid in tqdm(FoSs):
            dic  = read_file(os.path.join(FoS2Vec_path, fid))
            vecs = get_vec_func(dic, 1990, 2018)
            if len(vecs) >= thresh:    
                # 实验组
                mer_same_dis[fid] = maximum_explainable_variance_func(vecs)
                # 对照组
                shuffle_index = np.arange(len(vecs_2))
                random.shuffle(shuffle_index)
                control_size = min(len(vecs), len(vecs_2))
                shuffle_index = shuffle_index[:control_size]
                mer_diff_dis[fid] = maximum_explainable_variance_func(vecs_2[shuffle_index, :])
        save_file(mer_same_dis, path_same)
        save_file(mer_diff_dis, path_diff)

    # 绘制分布
    x = list(mer_diff_dis.keys())
    y_diff = np.array([mer_diff_dis[i] for i in x])  # control group - random sampling
    y_same = np.array([mer_same_dis[i] for i in x])  # test group
    y_sub  = y_same - y_diff

    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    ax1 = sns.distplot(y_same, hist=True, kde=True, rug=False,
                       bins=15,
                       # fit=norm,
                       hist_kws = {'rwidth':1, 'color':'lime', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.75, "label": r"$MEV(V_j(:t))$"},
                       kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": ""},
                       #rug_kws  = {"color": "black",   "alpha":0.25, "linewidth": 0.01, "height":0.05},
                       #fit_kws  = {"color": "black", "alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": "Normal"},
                       )
    sns.distplot(y_sub, hist=True, kde=True, rug=False,
                 bins=15,
                 # fit=norm,
                 hist_kws = {'rwidth':1, 'color':'blue', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.5, "label": r"$MEV(V_j(:t))$ - $MEV(V_j(:t), \widehat{V_j(:t)})$"},
                 kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": "KDE"},
                 #rug_kws  = {"color": "black",  "alpha":0.25, "linewidth": 0.01, "height":0.05},
                 #fit_kws  = {"color": "black", "alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": ""}
                 ax=ax1)

    ax2 = plt.twinx()
    sns.distplot(y_diff, hist=True, kde=True, rug=False,
                 bins=15,
                 # fit=norm,
                 hist_kws = {'rwidth':1, 'color':'gray', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.75, "label": r"$MEV(V_j(:t), \widehat{V_j(:t)})$"},
                 kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": ""},
                 #rug_kws  = {"color": "black", "alpha":0.25, "linewidth": 0.01, "height":0.05},
                 #fit_kws  = {"color": "black","alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": ""},
                 ax=ax2)
    
    ax1.set_xlabel(r"Maximum explainable variance ($t$=2018)")
    ax1.set_ylabel("Density")
    ax2.set_ylabel("")
    ax1.legend(loc='upper left', frameon=False, fontsize=15)
    ax2.legend(loc='upper right', frameon=False, fontsize=15)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_xlim(0, 1)
    ax1.set_yticks(np.arange(0, 35, 5))
    ax2.set_yticks(np.arange(0, 175, 25))
    ax2.tick_params(axis='y',colors='gray')
    

def check_for_anisotropy_from_self_distance(sampling_size_1, max_mp_num,
                                      min_sampling_times, max_sampling_times, sampling_discount_ratio):
    """ demostrate the anisotropy in the topic embeddings """
    # 说明 存在anisotropy
    # 说明 相同FoS的向量距离 < 不同FoS的向量距离

    path_same = "./temp/self_dis_same_dis.pkl"
    path_diff = "./temp/self_dis_diff_dis.pkl"
    path_rand = "./temp/vecs_2.pkl" # 随机向量池
    
    if os.path.exists(path_same) and os.path.exists(path_diff):
        self_dis_same_dis = read_file(path_same)
        self_dis_diff_dis = read_file(path_diff)
    else:
        if os.path.exists(path_rand):
            vecs_2 = read_file(path_rand)
        else:
            vecs_2 = sampling_control_group(1990, 2018)
            
        thresh = 100
        self_dis_same_dis = dict()
        self_dis_diff_dis = dict()
        for fid in tqdm(FoSs):
            dic  = read_file(os.path.join(FoS2Vec_path, fid))
            vecs = get_vec_func(dic, 1990, 2018)
            if len(vecs) >= thresh:    
                self_dis_same_dis[fid] = calculate_self_distance_func(vecs, vecs,   True,  sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
                self_dis_diff_dis[fid] = calculate_self_distance_func(vecs, vecs_2, False, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
        save_file(self_dis_same_dis, path_same)
        save_file(self_dis_diff_dis, path_diff)

    # 绘制分布
    x = list(self_dis_diff_dis.keys())
    y_diff = 1 / np.array([self_dis_diff_dis[i] for i in x])
    y_same = 1 / np.array([self_dis_same_dis[i] for i in x])
    y_sub  = y_same - y_diff 
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    ax1 = sns.distplot(y_same, hist=True, kde=True, rug=False,
             bins=15,
             # fit=norm,
             hist_kws = {'rwidth':1, 'color':'lime', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.75, "label": "$SDIS(V_j(:t))$"},
             kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": ""},
             # rug_kws  = {"color": "black",   "alpha":0.25, "linewidth": 0.01, "height":0.05},
             # fit_kws  = {"color": "black", "alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": "Normal"},
             )
    sns.distplot(y_sub, hist=True, kde=True, rug=False,
             bins=15,
             # fit=norm,
             hist_kws = {'rwidth':1, 'color':'blue', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.5, "label": r"$SDIS(V_j(:t), \widehat{V_j(:t)})$ - $SDIS(V_j(:t))$"},
             kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": "KDE"},
             # rug_kws  = {"color": "black",  "alpha":0.25, "linewidth": 0.01, "height":0.05},
             # fit_kws  = {"color": "black", "alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": ""},
             ax = ax1)
    ax2 = plt.twinx()
    sns.distplot(y_diff, hist=True, kde=True, rug=False,
                 bins=15,
                 # fit=norm,
                 hist_kws = {'rwidth':1, 'color':'gray', "edgecolor":"white", "histtype": "bar", 'linewidth':1, 'alpha':0.75, "label": r"$SDIS(V_j(:t), \widehat{V_j(:t)})$"},
                 kde_kws  = {"color": "red", "alpha":0.5, "linewidth": 3,    "shade":False, "label": ""},
                 # rug_kws  = {"color": "black", "alpha":0.25, "linewidth": 0.01, "height":0.05},
                 # fit_kws  = {"color": "black","alpha": 0.25, "linewidth": 2, "linestyle": "--", "label": ""},
                 ax=ax2)
 
    
    ax1.set_xlabel(r"Self distance ($t$=2018)")
    ax1.set_ylabel("Density")
    ax2.set_ylabel("")
    ax1.legend(loc='upper left', frameon=False, fontsize=15)
    ax2.legend(loc='upper right', frameon=False, fontsize=15)
    ax1.set_xticks(np.arange(0, 0.5, 0.1))
    ax1.set_yticks(np.arange(0, 35, 5))
    ax2.set_yticks(np.arange(0, 120, 20))
    ax2.tick_params(axis='y',colors='gray')


def results_for_VectorQuality():
    """ Experiments and results in Section 5.1 """
    check_for_anisotropy_from_maximum_explainable_variance()
    check_for_anisotropy_from_self_similarity(1000, 7, 10, 1000, 1)
    check_for_anisotropy_from_self_distance(3000, 7, 10, 1000, 1)


#%%
"""
    Experiments and results in Section 5.2
    Analyze the evolution pattern of topics' semantic consistency 
    (IM, DM, U-shape, Inverted U-shape)
"""

def fit_quadratic_curve(time, y):
    
    def quadratic_curve(x, a, b, c):
        y = a * x**2 + b * x + c
        return y
    
    x = (np.array(time) - min(time)) / (max(time) - min(time))
    popt, pcov = curve_fit(quadratic_curve, x, y)
    y_hat = quadratic_curve(x, *popt)
    r2 = r2_score(y, y_hat)
    return popt, r2, y_hat     


def plot_self_similarity(time, ssim, freq, fid, fit=True):
    """
    time: 时间; ssim: 逐年累计self similarity; 逐年累计频率; fos name
    """
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 22
              }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False 
    
    freq = np.array(freq)
    s = freq / np.sum(freq)
    s = np.minimum(np.maximum(s * 3e3, 5), 500)
    
    plt.plot(time, ssim, label=fid[:-4], c='gray')
    plt.scatter(time, ssim, color='red', s=s, marker='s', alpha=0.5)
    freqTexts = list()
    for x, y, f in zip(time, ssim, freq):
        freqText = plt.text(x, y, f, fontsize=15, color = "black", 
                            weight = "light", verticalalignment='baseline', 
                            horizontalalignment='right', rotation=0)
        freqTexts.append(freqText)
    adjust_text(freqTexts, )
        
    if fit:
        popt, r2, y_hat = fit_quadratic_curve(time, ssim)
        plt.plot(time, y_hat, label=r"$R^2$={:.4f}".format(r2), c='black', linestyle='--', linewidth=3)    
        
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Self-similarity")
    plt.xticks(np.arange(min(time), max(time)+1, 1), rotation=45, fontsize=20)


def plot_maximum_explainable_variance(time, mer, freq, fid, fit=True):
    """
    time: 时间; mer: 逐年累计maximum_explainable_variance; 逐年累计频率; fos name
    """
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 22
              }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False 
    
    freq = np.array(freq)
    s = freq / np.sum(freq)
    s = np.minimum(np.maximum(s * 5e3, 3), 500)
    
    plt.plot(time, mer, label=fid[:-4], c='gray')
    plt.scatter(time, mer, color='orange', s=s, marker='o', alpha=0.5)
    freqTexts = list()
    for x, y, f in zip(time, mer, freq):
        freqText = plt.text(x, y, f, fontsize=15, color = "black", 
                            weight = "light", verticalalignment='baseline', 
                            horizontalalignment='right', rotation=0)
        freqTexts.append(freqText)
    if fit:
        popt, r2, y_hat = fit_quadratic_curve(time, mer)
        plt.plot(time, y_hat, label=r"$R^2$={:.4f}".format(r2), c='black', linestyle='--', linewidth=3)    
        
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Maximum Explainable Variance")
    plt.xticks(np.arange(min(time), max(time)+1, 1), rotation=45, fontsize=20)


def plot_self_distance(time, dis, freq, fid, fit=True):
    """
    time: 时间; ssim: 逐年累计self similarity; 逐年累计频率; fos name
    """
    dis = 1 / np.array(dis)
    
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 22
              }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False 
    
    freq = np.array(freq)
    s = freq / np.sum(freq)
    s = np.minimum(np.maximum(s * 3e3, 5), 500)
    
    plt.plot(time, dis, label=fid[:-4], c='gray')
    plt.scatter(time, dis, color='green', s=s, marker='p', alpha=0.5)
    freqTexts = list()
    for x, y, f in zip(time, dis, freq):
        freqText = plt.text(x, y, f, fontsize=15, color = "black", 
                            weight = "light", verticalalignment='baseline', 
                            horizontalalignment='right', rotation=0)
        freqTexts.append(freqText)
    
    if fit:
        popt, r2, y_hat = fit_quadratic_curve(time, dis)
        plt.plot(time, y_hat, label=r"$R^2$={:.4f}".format(r2), c='black', linestyle='--', linewidth=3)    
        
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("1 / Self-distance")
    plt.xticks(np.arange(min(time), max(time)+1, 1), rotation=45, fontsize=20)


def plot_all_metrics(time, ssim, mer, sdis, freq, fid, 
                     ssim_=[], mer_=[], sdis_=[], fit=True, normalized=True, cls_type=''):

    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 12
              }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False 

    sdis  = 1 / np.array(sdis)
    sdis_ = 1 / np.array(sdis_)
    
    ssim_normalized = (ssim - min(ssim)) / (max(ssim) - min(ssim))
    mer_normalized  = (mer - min(mer))   / (max(mer) - min(mer))
    sdis_normalized = (sdis - min(sdis)) / (max(sdis) - min(sdis))
    
    ssim_normalized = (ssim - np.mean(ssim)) / np.std(ssim)
    mer_normalized  = (mer - np.mean(mer))   / np.std(mer)
    sdis_normalized = (sdis - np.mean(sdis)) / np.std(sdis)
    avg_metric = (ssim_normalized + mer_normalized + sdis_normalized) / 3
    
    Y = [ssim,  mer,  sdis, avg_metric]
    Y_= [ssim_, mer_, sdis_, []]
    C = ["red", "blue", "green", "black"]
    S = ["s", "o", "p", "+"]
    Ylabel = [r"$SSIM$", r"$MEV$", r"$SDIS^{-1}$", r"$1/3*(SSIM+MEV+SDIS^{-1}$)"]
    
    for i in range(len(Y)):
        ax = plt.subplot(2, 2, i+1)
        if not normalized:
            Y_i  = np.array(Y[i])
        else:
            Y_i  = np.array(Y[i])
            Y_i  = (Y_i - min(Y_i))   / (max(Y_i) - min(Y_i))
        ax.plot(time, Y_i, linewidth=1, linestyle='--', c=C[i], marker=S[i], markersize=4)
        
        # 轴刻度
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        up_y   = math.ceil(max(Y_i) * 1000)
        down_y = math.floor(min(Y_i) * 1000)
        yticks = np.linspace(down_y, up_y, 5) / 1000
        ax.set_yticks(yticks)
        # ax.set_ylim(yticks[0], yticks[-1])
        if min(time) % 10 >= 5:
            down_x = min(time) // 10 * 10 + 5
        else:
            down_x = min(time) // 10 * 10
        up_x = 2025
        xticks = np.arange(down_x, up_x, 5)
        ax.set_xticks(xticks)
        
        # 对照组
        if len(Y_[i]) != 0:
            if not normalized:
                Y_i_ = np.array(Y_[i])
            else:
                Y_i_ = np.array(Y_[i])
                Y_i_ = (Y_i_ - min(Y_i_)) / (max(Y_i_) - min(Y_i_))
            ax.plot(time,    Y_i_, c='gray', linewidth=1, linestyle='--')
            ax.scatter(time, Y_i_, color='gray', s=25, marker=S[i], alpha=1)
        ax.set_xlabel("Time")
        ax.set_ylabel(Ylabel[i])
        
        # 二次函数拟合
        if fit and i == 3:
            popt, r2, y_hat = fit_quadratic_curve(time, Y[i])
            ax.plot(time, y_hat, label=r"${:.3f}t^2+{:.3f}t+{:.3f}$".format(popt[0], popt[1], popt[2]) +"\n" + r"$R^2={:.3f}$".format(r2),
                    c='gray', linestyle='-.', linewidth=1)    
            ax.legend(frameon=False, fontsize=11)  
            ax.set_xticks(xticks)
    plt.suptitle(fid[:-4] + " ({})".format(cls_type), fontsize=20, fontweight='bold')
    plt.tight_layout()

    
def calculate_metrics(dic, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio):
    """
        calculate original metics
        Self-similarity; Self-distance; Maximum explainable variance
    """
    year_list  = dic.keys()
    year_list  = sorted(year_list)
    start_year = max(min(year_list), 1990)
    end_year   = min(max(year_list), 2018)
    
    ssim = list()  # self-similarity
    mer  = list()  # maximum_explainable_variance
    sdis = list()  # self-distance
    freq = list()  # word frequency
    time = list()  # year
    for year in range(start_year, end_year + 1):
        vecs = get_vec_func(dic, start_year, year)
        if len(vecs) < 10:
            continue
        # 逐年计算语义一致性指标
        SSIM = calculate_self_similarity_func(vecs, vecs, True, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
        MER  = maximum_explainable_variance_func(vecs)
        SDIS = calculate_self_distance_func(vecs, vecs, True, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
        FREQ = len(vecs)
        
        ssim.append(SSIM)
        mer.append(MER)
        sdis.append(SDIS)
        freq.append(FREQ)
        time.append(year)
    return ssim, mer, sdis, freq, time


def calculate_metrics_random(dic, vecs_2, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio):
    """
        calculate random metics
    """
    year_list  = dic.keys()
    year_list  = sorted(year_list)
    start_year = max(min(year_list), 1990)
    end_year   = min(max(year_list), 2018)
    
    ssim = list()  # self-similarity
    mer  = list()  # maximum_explainable_variance
    sdis = list()  # self-distance
    freq = list()  # word frequency
    time = list()  # year
    for year in range(start_year, end_year + 1):
        vecs = get_vec_func(dic, start_year, year)
        if len(vecs) < 10:
            continue
        # 获取主题向量池
        shuffle_index = np.arange(len(vecs))
        random.shuffle(shuffle_index)
        control_size = min(len(vecs), 10000)
        shuffle_index = shuffle_index[:control_size]
        vecs_1_part = vecs[shuffle_index, :]
        
        # 获取等量随机向量池
        shuffle_index = np.arange(len(vecs_2))
        random.shuffle(shuffle_index)
        control_size = min(min(len(vecs), len(vecs_2)), 5000)
        shuffle_index = shuffle_index[:control_size]
        vecs_2_part = vecs_2[shuffle_index, :]
        
        # 逐年计算语义一致性指标-基于随机向量组
        SSIM = calculate_self_similarity_func(vecs_1_part, vecs_2_part, False, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
        MER  = maximum_explainable_variance_func(vecs_2_part)
        SDIS = calculate_self_distance_func(vecs_1_part, vecs_2_part, False, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
        FREQ = len(vecs)
        del vecs_1_part
        del vecs_2_part
        
        ssim.append(SSIM)
        mer.append(MER)
        sdis.append(SDIS)
        freq.append(FREQ)
        time.append(year)
    return ssim, mer, sdis, freq, time


def calculate_metrics_shuffle(dic, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio):
    """ shuffle embeddings, calculate ssim, mer, sdis"""
    # 该主题所有向量构成的向量池
    all_vecs  = list()  
    for year in dic:
        all_vecs += dic[year]    
    random.shuffle(all_vecs)
    # 类内乱序
    dic_shuffle = dict()
    for year in dic:
        sample_num = len(dic[year])
        vecs = random.sample(all_vecs, sample_num)
        dic_shuffle[year] = vecs
    # 年份洗牌后, 计算语义一致性指标
    ssim, mer, sdis, freq, time = calculate_metrics(dic_shuffle, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
    return ssim, mer, sdis, freq, time


def plot_3d_func(fos, results, FoS2Vec_path, cls_type=''):
    """逐年散点3d图"""
    dic  = read_file(os.path.join(FoS2Vec_path, fos))  # fos的向量字典
    _, _, _, _, time = results[fos]
    beg_year = min(time)                               # fos的启始年
    end_year = max(time)                               # fos的最后一年 2018
    vecs = get_vec_func(dic, 1990, 2018)               # 1990 - 2018 所有向量
    pca = decomposition.PCA(n_components=3)            # pca 
    pca.fit(vecs)
    vecs_3d_list = dict()
    for year in range(beg_year, end_year + 1):
        vecs_i = get_vec_func(dic, year, year)
        if len(vecs_i) != 0:    
            vecs_i_3d = pca.transform(vecs_i)
            vecs_3d_list[year] = vecs_i_3d
        else:
            vecs_3d_list[year] = []
    dates  = np.arange(beg_year, end_year + 1)         # Time
    colors = np.linspace(-1, 1, len(dates))            # color bar 

    # 绘图
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 10
              }
    rcParams.update(config)
    ax = fig.add_subplot(111, projection='3d')
    
    xyzcs = list()  # (x, y, z, color)
    cents = list()
    for i, year in enumerate(range(beg_year, end_year + 1)):
        vecs_i_3d = vecs_3d_list[year]
        if len(vecs_i_3d) == 0:
            continue
        
        samples_size = min(len(vecs_i_3d), 100)
        print("{}/{}".format(samples_size, len(vecs_i_3d)))
        
        vecs_i_3d = random.sample(list(vecs_i_3d), samples_size)
        vecs_i_3d = np.array(vecs_i_3d)
        
        xs = vecs_i_3d[:, 0]
        ys = vecs_i_3d[:, 1]
        zs = vecs_i_3d[:, 2]
        cs = colors[i] * np.ones(len(xs))
        xyzcs.append(np.array([xs, ys, zs, cs]))
        cents.append(np.array([np.mean(xs), np.mean(ys), np.mean(zs), colors[i]]))
        # break
    
    cmap = 'seismic'
    cmap2= "Greens"
    xyzcs = np.concatenate(xyzcs, axis=-1)  
    ax.scatter(xyzcs[0, :], xyzcs[1, :], xyzcs[2, :], c=xyzcs[3, :], cmap=cmap, s=2, alpha=0.4)
         
    # cents = np.array(cents)
    # points = cents[:, :3]
    # points = points.reshape(-1, 1, 3)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = Line3DCollection(segments, cmap=plt.get_cmap(cmap2), linestyles='dashed')
    # lc.set_array(cents[:, 3])  # 设置颜色
    # lc.set_linewidth(1.5)
    # ax.add_collection3d(lc)
    # ax.scatter(cents[:, 0], cents[:, 1], cents[:, 2], c=cents[:, 3], cmap=cmap2, s=15, marker='<') 
    # ax.legend(frameon=False)
    
    # 轴刻度
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_x   = math.ceil(max(xyzcs[0, :]) * 1000)
    down_x = math.floor(min(xyzcs[0, :]) * 1000)
    xticks = np.linspace(down_x, up_x, 5) / 1000
    ax.set_xticks(xticks)
    
    up_y   = math.ceil(max(xyzcs[1, :]) * 1000)
    down_y = math.floor(min(xyzcs[1, :]) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    ax.set_yticks(yticks)
    
    up_z   = math.ceil(max(xyzcs[2, :]) * 1000)
    down_z = math.floor(min(xyzcs[2, :]) * 1000)
    zticks = np.linspace(down_z, up_z, 5) / 1000
    ax.set_zticks(zticks)
    plt.title(fos[:-4] + " ({})".format(cls_type), fontsize=20, fontweight='bold')
    
    # color bar
    labels = [str(i) for i in dates]
    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=ax,
                     ticks=colors, fraction=0.027, pad=0.02, shrink=1.0)
    cb.ax.set_yticklabels(labels)
    plt.show()


def results_for_EvolutionTrends():
    """ Experiments and results in Section 4.2
        Four general evolution pattern of topic consistency is identified based on the K-Means algorithm.
    """
    """ 逐年计算主题的语义一致性指标 """
    
    thresh = 100
    sampling_size_1 = 1000
    max_mp_num = 7
    max_sampling_times = 100
    min_sampling_times = 20
    sampling_discount_ratio = 0.5
    
    results_file_path = "./temp/results_for_evolutionary_trajectory.pkl"
    random_file_path  = "./temp/results_for_evolutionary_trajectory(random).pkl"
    path_rand         = "./temp/vecs_2.pkl" # 随机向量池
    
    # (1)计算原指标
    if os.path.exists(results_file_path):
        results = read_file(results_file_path)
        # 检查是否所有主题的指标被全部计算, 否则补算
        for fid in tqdm(FoSs):
            if fid in results:
                continue 
            dic  = read_file(os.path.join(FoS2Vec_path, fid))
            vecs = get_vec_func(dic, 1990, 2018)
            if len(vecs) >= thresh:
                (ssim, mer, sdis, freq, time) = calculate_metrics(dic, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
                results[fid] = (ssim, mer, sdis, freq, time)
        save_file(results, results_file_path)
    else:
        # 计算指标
        results = dict()
        for fos in tqdm(FoSs):
            dic  = read_file(os.path.join(FoS2Vec_path, fos))
            vecs = get_vec_func(dic, 1990, 2018)
            if len(vecs) >= thresh:
               ssim, mer, sdis, freq, time = calculate_metrics(dic, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
               results[fos] = (ssim, mer, sdis, freq, time)
        save_file(results, results_file_path)
    
    # 计算随机指标 (运算速度慢)
    if os.path.exists(path_rand):
       vecs_2 = read_file(path_rand)
    else:
       vecs_2 = sampling_control_group(1990, 2018)
    # 
    if os.path.exists(random_file_path):
        # 检查是否所有主题的指标被全部计算, 否则补算
        results_random = read_file(random_file_path)
        count = 0
        for fid in tqdm(FoSs):
            count += 1
            if fid in results_random:
                continue 
            dic  = read_file(os.path.join(FoS2Vec_path, fid))
            vecs = get_vec_func(dic, 1990, 2018)
            if len(vecs) >= thresh:
                ssim, mer, sdis, freq, time = calculate_metrics_random(dic, vecs_2, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
                results_random[fid] = (ssim, mer, sdis, freq, time)
            # 每计算100个主题, 储存/次
            if count >= 100:
                save_file(results_random, random_file_path)
                count = 0
        save_file(results_random, random_file_path)
    else:
        results_random = dict()
        count = 0
        for fos in tqdm(FoSs):
            count += 1
            dic  = read_file(os.path.join(FoS2Vec_path, fos))
            vecs = get_vec_func(dic, 1990, 2018)
            if len(vecs) >= thresh:
               ssim, mer, sdis, freq, time = calculate_metrics_random(dic, vecs_2, sampling_size_1, max_mp_num, min_sampling_times, max_sampling_times, sampling_discount_ratio)
               results_random[fos] = (ssim, mer, sdis, freq, time)       
            if count >= 100:
                save_file(results_random, random_file_path)
                count = 0 
        save_file(results_random, random_file_path)
    
    # 读取随机指标
    path_ssim_diff = "./temp/self_sim_diff_dis.pkl"
    path_mer_diff  = "./temp/mer_diff_dis.pkl"
    path_sdis_diff = "./temp/self_dis_diff_dis.pkl"
    path_anisotropy_adjusted_res ="./temp/anisotropy_adjusted_res.pkl"
    
    ssim_diff = read_file(path_ssim_diff)
    mer_diff  = read_file(path_mer_diff)
    sdis_diff = read_file(path_sdis_diff)
    
    # exlcude anisotropy
    anisotropy_adjusted_res = dict()
    for fid in tqdm(results):
        ssim, mer, sdis, freq, time = results[fid]
        ssim = list(np.array(ssim)   - ssim_diff[fid])
        mer  = list(np.array(mer)    - mer_diff[fid])
        sdis = list(1/np.array(sdis) - 1/sdis_diff[fid])
        anisotropy_adjusted_res[fid] = (ssim, mer, sdis, freq, time)
    save_file(anisotropy_adjusted_res, path_anisotropy_adjusted_res)
    
    
    """
    K-Means: 时间序列数据聚类分析
    """
    anisotropy_adjusted_res = read_file(path_anisotropy_adjusted_res)
    # 生成tslearn需要的时间序列数据输入, 长度不同的时期序列使用nan补齐
    time_len = 5
    fids = list()
    X_bias = list()
    for fid in anisotropy_adjusted_res:
        ssim, mer, sdis, freq, time = anisotropy_adjusted_res[fid]
        if len(time) >= time_len:
            ts_fid = np.array([ssim, mer, sdis]).T
            fids.append(fid)
            X_bias.append(ts_fid)
    X_bias = tslearn.utils.to_time_series_dataset(X_bias)
    # 标准化处理 - 聚类的目标是识别形状 (compare shapes in an amplitude-invariant manner)
    X_bias = TimeSeriesScalerMeanVariance(mu=0, std=1).fit_transform(X_bias)
 
    # 聚类, dynamic time warping (dtw) - 时间调整对齐距离
    n_clusters = 4
    path_kmeans_res = './temp/TimeSeriesKMeans(km={}).pkl'.format(n_clusters)
    if not os.path.exists(path_kmeans_res):
        km = TimeSeriesKMeans(n_clusters=n_clusters, max_iter=10, metric='dtw', n_jobs=6)
        km.fit(X_bias)
        save_file(km, path_kmeans_res)  # 存放模型
    else:
        km = read_file(path_kmeans_res) # 读取模型
    Y_label = km.predict(X_bias)
    # km.transform(X_bias)
    
    # 聚类效果分析 - 质心
    fig = plt.figure(figsize=(8, 8)) # 10, 8
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 15
              }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False 
    
    for i in range(n_clusters):
        if n_clusters == 4:
            ax = plt.subplot(2, 2, i+1)
            cluster_type = {0: 0, 1: 3, 2: 2, 3: 1}
            cluster_name = ["IM", "DM", "U-shape",  "Inverted U-shape"]
            
        if n_clusters == 5:
            ax = plt.subplot(3, 2, i+1)
            cluster_type = {0: 0, 1: 1, 2: 2, 3: 3, 4:4}
            cluster_name = ["Inverted U-shape", "U-shape", "IM",  "DM", "IM or Inverted U-shape"]
        
        if n_clusters == 6:
            ax = plt.subplot(3, 2, i+1)
            cluster_type = {0: 0, 1: 1, 2: 2, 3: 3, 4:4, 5:5}
            cluster_name = ["DM", "U-shape or IM", "U-shape",  "Inverted U-shape", "IM", "Inverted U-shape"]
        
        yi = cluster_type[i]
        
        for xx in X_bias[Y_label == yi]:
            ax.plot(np.mean(xx, axis=-1), alpha=.02, c='gray')
        class_num = len(X_bias[Y_label == yi])
        total_num = len(X_bias)
        class_r   = class_num / total_num
        
        centroid_yi = km.cluster_centers_[yi]
        # pca = PCA(n_components=1)
        # centroid_yi_ = pca.fit_transform(centroid_yi)  # pca可能改变单调性
        
        # 类别的质心
        centroid_yi_mu = np.mean(centroid_yi, axis=-1)
        centroid_yi_mu_smooth = np.convolve(centroid_yi_mu, np.ones(1) / 1,'vlid')
        plt.plot(centroid_yi_mu_smooth, c="r", linestyle="--", linewidth=2)
        
        # 二次函数拟合质心
        # popt, r2, y_hat = fit_quadratic_curve(np.arange(len(centroid_yi_mu)), centroid_yi_mu)
        # plt.plot(y_hat, c="blue", linestyle="dotted", linewidth=2)
        
        ax.set_title(cluster_name[i])
        
        plt.text(0.5, 0.5, "{}\n{:.2f}%".format(class_num, 100 * class_r), 
                 transform=plt.gca().transAxes, color='black', fontsize=20)

        plt.ylim(-3, 3)
        plt.xticks(np.arange(0, 35, 5))
        plt.xlim(0, 30)
        plt.xlabel("Time")
        plt.ylabel(r"$1/3(SSIM+MEV+SDIS)$")
    plt.tight_layout()

    """
    Case study 案例分析
    """
    fid2AVG = dict()      # 三个指标normalized, 然而取平均
    km_results = dict()  # 
    for yi, xi, fid in zip(Y_label, X_bias, fids):
        fid2AVG[fid] = np.mean(xi, axis=-1)
        if yi not in km_results:
            km_results[yi] = dict()
            km_results[yi]["samples"] = list()
            km_results[yi]["cluster_centers_"] =  km.cluster_centers_[yi]
        km_results[yi]["samples"].append(fid)
        
    # 倒U型, 上升类, 下降类, 正U型
    yi = 0
    fos = random.sample(km_results[yi]["samples"], 1)[0]
    (ssim, mer, sdis, freq, time) = anisotropy_adjusted_res[fos]
    plot_all_metrics(time, ssim, mer, sdis, freq, fos, [], [], [], fit=True, normalized=False)
  
    """ 
    可视化分析 
    """
    # FoSs = list(anisotropy_adjusted_res.keys())
    # fos  = random.sample(FoSs, 1)[0]
    # class_idx = 0
    # fos  = random.sample(km_results[class_idx]['samples'], 1)[0]
    fos  = "semantic network.pkl".lower()
    (ssim, mer, sdis, freq, time) = anisotropy_adjusted_res[fos]
    plot_all_metrics(time, ssim, mer, sdis, freq, fos, fit=True, normalized=False, cls_type='Inverted U-shape') # Inverted U-shape
    plot_3d_func(fos, results, FoS2Vec_path, cls_type='Inverted U-shape')
    

#%%
"""
    Experiments and results in Section 5.3
    Calculate others similarity & others distance
"""

def calculate_centriod_dis(Centroid, fos_i, fos_j):
    """计算不同fos质心的差距"""
    tmp = dict()
    Centroid_i   = Centroid[fos_i]  # 质心i
    Centroid_j   = Centroid[fos_j]  # 质心j
    start_year_i = min(Centroid_i.keys())
    start_year_j = min(Centroid_j.keys())
    end_year_i   = max(Centroid_i.keys())
    end_year_j   = max(Centroid_j.keys())
    start_year   = max(start_year_i, start_year_j)
    end_year     = 2018
    for year in range(start_year, end_year + 1):
        # 计算质心距离
        if year in Centroid[fos_i]:
            vecs_i_avg = Centroid[fos_i][year]
        else:
            vecs_i_avg = Centroid[fos_i][end_year_i]
        if year in Centroid[fos_j]:
            vecs_j_avg = Centroid[fos_j][year]
        else:
            vecs_j_avg = Centroid[fos_j][end_year_j]
        vecs_i_avg = torch.tensor(vecs_i_avg).float()
        vecs_j_avg = torch.tensor(vecs_j_avg).float()
        cos_ij     = torch.cosine_similarity(vecs_i_avg, vecs_j_avg).numpy()[0]
        dis_ij     = torch.dist(vecs_i_avg, vecs_j_avg).numpy()
        dis_ij     = np.array([dis_ij])[0]
        
        cos_ij = np.array(cos_ij, dtype=np.float16)
        dis_ij = np.array(dis_ij, dtype=np.float16)
        tmp[year] = np.array([cos_ij, dis_ij])
    return tmp


def calculate_centriod_dis_MP(Centroid, cofos_list_i):
    tmp2 = dict()
    for name_ij in cofos_list_i:
        fos_i, fos_j = name_ij.split(";")
        tmp = calculate_centriod_dis(Centroid, int(fos_i), int(fos_j))
        tmp2[name_ij] = tmp
    return tmp2


def plot_intra_distance(ssim, osim, time, fos, xticks=[], ax1_yticks=[], ax2_yticks=[]):
                        
    """
    (ssim, time): self distance (inter similarity)
    (osim, time): intra similarity
    """
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    plt.rcParams['axes.unicode_minus'] = False 
    
    Y = [ssim,  osim]
    C = ["red", "blue", "green", "brown"]
    S = ["s", "o", "+", "p"]
    Ylabel = ["SSIM", "OSIM"]  # 类内相似性, 类间相似性

    ax1 = fig.add_subplot(111)
    ax1.plot(time, ssim, c='red', linewidth=1, linestyle='--', marker="s", label="")
    ax2 = plt.twinx()
    ax2.plot(time, osim, c='blue', linewidth=1, linestyle='--', marker="o", label="")
    
    ax1.set_ylabel("SSIM", color='red')
    ax2.set_ylabel("OSIM", color='blue')
    ax1.set_xlabel("Time")
    ax1.legend(frameon=False, loc='upper left')
    ax2.legend(frameon=False, loc='upper right')
    plt.title(fos[:-4], fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # 坐标轴刻度
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_y   = math.ceil(max(ssim) * 1000)
    down_y = math.floor(min(ssim) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    ax1.set_yticks(yticks)
    
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    up_y   = math.ceil(max(osim) * 1000)
    down_y = math.floor(min(osim) * 1000)
    yticks = np.linspace(down_y, up_y, 5) / 1000
    ax2.set_yticks(yticks)
    
    if min(time) % 10 >= 5:
        down_x = min(time) // 10 * 10 + 5
    else:
        down_x = min(time) // 10 * 10
    up_x = 2025
    xticks = np.arange(down_x, up_x, 5)
    ax1.set_xticks(xticks)
    ax1.tick_params(axis='y',colors='red')
    ax2.tick_params(axis='y',colors='blue')


def results_for_analyze_evolutionary_trajectory_Step2():
    """获取fos_i与fos_j之间的距离信息"""
    
    FoS2Vec_path = os.path.join(abs_file_path, "FoS2Vec")
    results = read_file("./temp/results_for_evolutionary_trajectory.pkl")
    FoSs = list(results.keys())
    
    # 统一计算质心
    if not os.path.exists("./temp/Centroid.pkl"):
        Centroid = dict()
        for fos in tqdm(FoSs):
            Centroid[fos] = dict()
            dic = read_file(os.path.join(FoS2Vec_path, fos))
            start_year = max(min(dic.keys()), 1990)
            end_year   = min(max(dic.keys()), 2018)
            for year in range(start_year, end_year + 1):
                vecs = get_vec_func(dic, start_year, year)
                if len(vecs) < 10:
                    continue
                else:
                    vecs_avg = np.mean(vecs, axis=0, keepdims=True)
                    if year not in Centroid[fos]:
                        Centroid[fos][year] = vecs_avg
        save_file(Centroid, "./temp/Centroid.pkl")
    else:
        # 计算 fos_i 和 fos_j 逐年的质心的距离
        Centroid = read_file("./temp/Centroid.pkl")
    
    # 给fos编号, 节约内存
    Centroid_ = dict()
    fos2idx   = dict()
    idx2fos   = dict()
    for i, fos in enumerate(Centroid):
        fos2idx[fos] = i
        idx2fos[i]   = fos
        Centroid_[i] = Centroid[fos]
    del Centroid
    save_file(idx2fos, "./temp/idx2fos.pkl")
    save_file(fos2idx, "./temp/fos2idx.pkl")
    
    # 统计需要计算的总(fos_i, fos_j) pair
    results_cofos_dis = dict()
    for fos_i in tqdm(FoSs):
        for fos_j in FoSs:
            if fos_i == fos_j:
                continue
            name_ij = str(fos2idx[fos_i]) + ";" + str(fos2idx[fos_j])
            name_ji = str(fos2idx[fos_j]) + ";" + str(fos2idx[fos_i])
            if name_ij not in results_cofos_dis and name_ji not in results_cofos_dis:
                results_cofos_dis[name_ij] = dict()
                
    # 将上述总 (fos_j, fos_j) pair 划分成 100 块
    chunck_num = 100
    cofos_list = list(results_cofos_dis.keys())
    del results_cofos_dis
    start_c = 0
    end_c   = 0
    c_size  = math.ceil(len(cofos_list) / chunck_num)
    for c in tqdm(range(chunck_num)):
        end_c = min(start_c + c_size, len(cofos_list))
        cofos_list_c = cofos_list[start_c: end_c]          # 取出一块多进程处理
        save_file(cofos_list_c, os.path.join(abs_file_path, "FoS2FoS/cofos_list_c_{}.pkl".format(c)))
        start_c = end_c
    del cofos_list
    
    # 每块 (fos_i, fos_j) pair 多进程计算质心距离 (cosine, Euclidean)
    for c in range(chunck_num):
        print("Processing {}".format(c))
        start_time = time_lib.perf_counter()
        
        cofos_list_c = read_file(os.path.join(abs_file_path, "FoS2FoS/cofos_list_c_{}.pkl".format(c)))
        # 开始多进程
        total_size = len(cofos_list_c)
        mp_num     = 7
        mp_size    = math.ceil(total_size / mp_num)
        start_idx  = 0
        end_idx    = 0
        tmp3 = list()
        pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
        for i in range(mp_num):
            end_idx = min(start_idx + mp_size, len(cofos_list_c))
            cofos_list_i = cofos_list_c[start_idx: end_idx]
            tmp3.append(pool.apply_async(calculate_centriod_dis_MP, (Centroid_, cofos_list_i,)))  
            start_idx = end_idx
        pool.close()
        pool.join()
        # 获取结果
        results_cofos_dis = dict()
        for tmp3_i in tmp3:
            res = tmp3_i.get()
            for name_ij in res:
                results_cofos_dis[name_ij] = res[name_ij]
        save_file(results_cofos_dis, os.path.join(abs_file_path, "FoS2FoS/results_cofos_dis_{}.pkl".format(c)))
        end_time = time_lib.perf_counter()
        
        print("耗时: {}".format(round(end_time - start_time)))
        
    # 清除中间文件
    for c in tqdm(range(chunck_num)):
        if os.path.exists(os.path.join(abs_file_path, "FoS2FoS/cofos_list_c_{}.pkl".format(c))):
            os.remove(os.path.join(abs_file_path, "FoS2FoS/cofos_list_c_{}.pkl".format(c)))
    
    # 稀疏矩阵储存 (逐年)
    row_num = len(idx2fos)    
    for c in tqdm(range(chunck_num)):
        results_cofos_dis = read_file(os.path.join(abs_file_path, "FoS2FoS/results_cofos_dis_{}.pkl".format(c)))
        # 新增信息
        Matrixs = dict()
        for year in range(1990, 2018 + 1):
            Matrix = np.zeros((row_num, row_num), dtype=np.float16)
            Matrixs[year] = Matrix        
        for name_ij in tqdm(results_cofos_dis):
            fos_i, fos_j = name_ij.split(";")
            fos_i = int(fos_i)
            fos_j = int(fos_j)
            for year in results_cofos_dis[name_ij]:
                Matrix = Matrixs[year]
                cos_ij, dis_ij = results_cofos_dis[name_ij][year]                
                # Matrix[fos_i, fos_j] = cos_ij               # *** OSIM ***
                Matrix[fos_i, fos_j] = dis_ij                 # *** ODIS ***
        # 补充
        for year in range(1990, 2018 + 1):
            Current_Matrix = Matrixs[year]                    # 非稀疏矩阵
            path_SMatrix   = os.path.join(abs_file_path, "FoS2SM/SM_{}.pkl".format(year))
            if not os.path.exists(path_SMatrix):
                Current_SMatrix = csr_matrix(Current_Matrix)  # 转换成稀疏矩阵
                save_file(Current_SMatrix, path_SMatrix)
            else:
                Former_SMatrix = read_file(path_SMatrix)
                Former_Matrix  = csr_matrix(Former_SMatrix).toarray()
                Former_Matrix += Current_Matrix
                Former_SMatrix = csr_matrix(Former_Matrix)
                save_file(Former_SMatrix, path_SMatrix)
                
                
    # (2) 研究主题自距离(ssim) 与 主题与其它主题的质心距离
    idx2fos = read_file("./temp/idx2fos.pkl")
    fos2idx = read_file("./temp/fos2idx.pkl")
    results = read_file("./temp/results_for_evolutionary_trajectory.pkl")
    ODIS_OSIM = dict()
    for fos in results:
        ODIS_OSIM[fos] = dict() 
    for year in tqdm(range(1990, 2018 + 1)):
        Matrix = read_file(os.path.join(abs_file_path, "FoS2SM/SM_{}.pkl".format(year)))
        Matrix = csr_matrix(Matrix).toarray()  # 是上三角矩阵
        Matrix = Matrix + Matrix.T             # 转换成距离对称矩阵                    
        for i in range(len(Matrix)): 
            fos = idx2fos[i]
            ssim, mer, sdis, freq, time = results[fos]
            fos_beg_year = min(time)
            fos_end_year = max(time)
            if fos_beg_year <= year and year <= fos_end_year:
                Matrix_row_i = Matrix[i]
                denominator  = max(np.sum(Matrix_row_i != 0), 1)
                numerator    = np.sum(Matrix_row_i)
                cos_avg      = numerator / denominator
                ODIS_OSIM[fos][year] = cos_avg
    # save_file(ODIS_OSIM, "./temp/results_for_evolutionary_trajectory2.pkl")  # osim
    # save_file(ODIS_OSIM, "./temp/results_for_evolutionary_trajectory3.pkl")  # odis
    
    
    # case study
    results  = read_file("./temp/anisotropy_adjusted_res.pkl")
    results2 = read_file("./temp/results_for_evolutionary_trajectory2.pkl")    # osim
    results3 = read_file("./temp/results_for_evolutionary_trajectory3.pkl")    # odis
    
    fos = list(results.keys())[1001]      
    fos = "semantic network.pkl"
    ssim, mer, sdis, freq, time = results[fos] 
    osim = results2[fos]
    osim = [osim[t] for t in time]
    plot_intra_distance(ssim, osim, time, fos)
                        

#%%
"""
    Experiments and results in Section 5.3
    Conduct regression analysis with fixed time and entity effects
"""

def calculate_metrics_pearsonr(Matrix, columns_name, title=""):
    ''' 计算上述所有指标的相关系数 '''
    # pd dataframe
    Data = pd.DataFrame(Matrix)
    # Data = (Data - Data.min()) / (Data.max() - Data.min())
    # 计算 pearsonr
    corr_matrix = Data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
    corr = corr_matrix.copy()
    
    # 绘制热力图
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman", # 
              "font.size" : 12
              }
    rcParams.update(config)
    # SimHei 字体符号不正常显示
    plt.rcParams['axes.unicode_minus'] = False 
    
    cmap = sns.diverging_palette(230, 0, 90, 60, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                vmin=-1, vmax=1, cbar_kws={'shrink': 1}, linewidths=5, square=True,
                xticklabels=columns_name, yticklabels=columns_name)
    plt.yticks(rotation=0) 
    plt.title(title, fontsize=20)


def calculate_gini(freq_list):
    # 计算GINI-index 2023-4-7
    freq_list  = np.array(freq_list)
    freq_ratio = freq_list / max(sum(freq_list), 1)
    gini = 1 - sum(freq_ratio ** 2) # gini系数, 越靠近0, 表示越纯
    return gini


def get_unique_num(aids, time1, time2):
    # 作者数目, 机构数目, 期刊数目 from time1 to time2
    unique_id = dict()
    for year in range(time1, time2 + 1):
        if year in aids:
            for aid in aids[year]:
                if aid not in unique_id:
                    unique_id[aid] = 0
                unique_id[aid] += aids[year][aid]
    unique_num = len(unique_id)
    freq_list  = [unique_id[Id] for Id in unique_id]
    gini = calculate_gini(freq_list)
    return unique_num, gini
        

def get_cc_num(ccs, time1, time2):
    # 引用次数
    cc_full_num = 0 
    cc_frac_num = 0
    for year in range(time1, time2 + 1):
        if year in ccs:
            cc_full_num += ccs[year][0]
            cc_frac_num += ccs[year][1]
    return cc_full_num, cc_frac_num


def get_freq_num(fids, time1, time2, fos):
    # 共现次数 (自身频率, 其余主题频率)
    self_freq = 0
    unique_co_fos = dict()
    for year in range(time1, time2 + 1):
        if year in fids:
            for fos_ in fids[year]:
                if fos_ == fos[:-4]:
                    self_freq += fids[year][fos_]
                else:
                    if fos_ not in unique_co_fos:
                        unique_co_fos[fos_] = fids[year][fos_]
                    else:
                        unique_co_fos[fos_] += fids[year][fos_]
    
    unique_fos_num  = len(unique_co_fos.keys()) # 共现主题种类
    other_freq_list = list()                    # 共现主题频次
    for fos_ in unique_co_fos:
        other_freq_list.append(unique_co_fos[fos_])
    other_freq_sum  = sum(other_freq_list)
    gini = calculate_gini(other_freq_list)      # gini index  
    return self_freq, other_freq_sum, unique_fos_num, gini         


def get_inter_metrics(results, time1, time2, fos):
    """inter-class"""
    ssim, mer, sdis, freq, time = results[fos]
    tmp = dict()
    for i, t in enumerate(time):
        tmp[t] = (ssim[i], mer[i], sdis[i])
    
    max_time = time[-1]
    time2_   = min(time2, max_time)
    ssim_t, mer_t, sids_t = tmp[time2_]
    time_gap = time2_ - time[0] + 1
    return ssim_t, mer_t, sids_t, time_gap


def get_intra_metrics(results2, time1, time2, fos):  
    """intra-class"""
    max_time = max(results2[fos].keys())
    time2_   = min(time2, max_time)
    osim_t   = results2[fos][time2_]
    return osim_t
      

def results_for_linear_regression_analysis(): 
    """ experiments and results in Section 4.3 """
    # 1990-2018年逐年ssim, mer, sdis (Total FoS Freq >= 100)
    path_anisotropy_adjusted_res ="./temp/anisotropy_adjusted_res.pkl"
    # results  = read_file("./temp/results_for_evolutionary_trajectory.pkl")
    results  = read_file(path_anisotropy_adjusted_res)
    results2 = read_file("./temp/results_for_evolutionary_trajectory2.pkl")  # OSIM
    results3 = read_file("./temp/results_for_evolutionary_trajectory3.pkl")  # ODIS

    # 准备线性回归数据
    begyear_all = 1990
    endyear_all = 2018
    data = dict()
    for fos in tqdm(results):
        data_fos = list()
        
        # 读取词向量信息
        # vec_path = os.path.join(absolute_path, "FoS2Vec/{}".format(fos))
        # vecs     = read_file(vec_path)
        # 读取作者信息
        aid_path = os.path.join(abs_file_path, "FoS2Info/{}/{}".format("FoS2Aid", fos))
        aids     = read_file(aid_path)
        # 读取机构信息
        oid_path = os.path.join(abs_file_path, "FoS2Info/{}/{}".format("FoS2Oid", fos))
        oids     = read_file(oid_path)
        # 读取期刊信息
        vid_path = os.path.join(abs_file_path, "FoS2Info/{}/{}".format("FoS2Vid", fos))
        vids     = read_file(vid_path)
        # 读取引用信息
        cc_path  = os.path.join(abs_file_path, "FoS2Info/{}/{}".format("FoS2CC", fos))
        if os.path.exists(cc_path):
            ccs  = read_file(cc_path)
        else:
            ccs  = dict()
        # 读取共现信息 (FoS)
        fid_path = os.path.join(abs_file_path, "FoS2Info/{}/{}".format("FoS2Fid", fos))
        fids     = read_file(fid_path)
        
        # 确定序列时刻范围:[begyear, endyear] (该时段已计算fos的ssim, mer, sdis指标)
        _, _, _, _, time = results[fos]
        begyear_fos = min(time)
        endyear_fos = max(time)
        begyear = max(begyear_all, begyear_fos)  # 保证已计算 ssim, mer, sdis
        endyear = min(endyear_all, endyear_fos)  # 保证已计算 ssim, mer, sdis
        for year in range(begyear, endyear + 1):
            # 控制变量
            aids_num, aids_gini = get_unique_num(aids, year, year) # 当年的作者数
            oids_num, oids_gini = get_unique_num(oids, year, year) # 当年的机构数
            vids_num, vids_gini = get_unique_num(vids, year, year) # 当年的期刊数
            cc_full_num, cc_frac_num = get_cc_num(ccs, year, year)         # 当年的引用数
            fos_freq, cofos_freq, cofos_num, cofos_gini = get_freq_num(fids, year, year, fos) # 当年的采纳数
            # 因变量 和 自变量
            ssim_t, mer_t, sdis_t, time_gap = get_inter_metrics(results, year, year, fos) # 因变量
            osim_t = get_intra_metrics(results2, year, year, fos)                         # 自变量 
            odis_t = get_intra_metrics(results3, year, year, fos)
        
            data_fos.append([ssim_t, mer_t, sdis_t,        # 因变量
                             osim_t, 1 / odis_t,           # 自变量
                             fos, time_gap, year,          # 个体效应, 时间效应 (相对时间, 绝对时间)
                             aids_num,  oids_num,   vids_num,              # 控制变量
                             aids_gini, oids_gini,  vids_gini,             # 控制变量 
                             fos_freq,  cofos_freq, cofos_num, cofos_gini, # 控制变量
                             cc_full_num, cc_frac_num])                    # 控制变量
        data[fos] = data_fos
    save_file(data, "./temp/data_reg.pkl")
    
    
    # (2) 开始线性回归
    data_reg = read_file("./temp/data_reg.pkl")
    # 准备回归数据
    Examples = data_reg
    data = list()
    for fos in Examples:
        data += data_reg[fos]
    data = pd.DataFrame(data)
    data.columns = ["SSIM", "MER", "SDIS",
                    "OSIM",  r"ODIS",
                    "FoS", "RT", "T",
                    "N_A", "N_O", "N_J",
                    "G_A", "G_O", "G_J", 
                    "Freq", "CoFreq", "N_Co", "G_Co",
                    "CC_full", "CC_frac"]
    RT = pd.Categorical(data.RT)
    data = data.set_index(['FoS', "RT"], drop=True)
    data['RT']   = RT
    data['OSIM2'] = data["OSIM"] ** 2
    data["LogFreq"] = np.log(np.maximum(data["Freq"], 1e-3))       # log 采纳频次 (论文数)
    data["LogCC"]   = np.log(np.maximum(data["CC_full"],   1e-3))  # log 被引数
    
    # 线性相关性检查
    columns_name = [r"$SSIM$", r"$MEV$", r"$SDIS$", 
                    r"$OSIM$", r"$ODIS$", 
                    r"$N^S$", r"$N^{AI}$", r"$N^{JC}$",
                    r"$N^{F}$", r"$N^{CF}$", r"$N^{UCF}}$",
                    r"$N^{CC}$",
                    r"$G^{S}$", r"$G^{AI}$", r"$G^{JC}$", r"$G^{CF}$"]
    data_corr = data[["SSIM", "MER", "SDIS", 
                      "OSIM", r"ODIS",
                      "N_A", "N_O", "N_J",
                      "Freq", "CoFreq", "N_Co", 
                      "CC_full",
                      "G_A", "G_O", "G_J", "G_Co"]]
    calculate_metrics_pearsonr(data_corr, columns_name)
    
    # data_corr = data[["LogFreq", "LogCC", "Freq", "CC_full"]]
    # corr_matrix = data_corr.corr()
    # print(corr_matrix)
    # variance_inflation_factor(data_corr.values, 1)
    
    # Table 4
    # 固定效应 - 双向效应
    exog_vars = ['OSIM']
    exog_vars = ['OSIM', "LogFreq"]
    exog_vars = ['OSIM', "LogFreq", "LogCC"]
    exog_vars = ['OSIM', "LogFreq", "LogCC", "G_A"]
    exog_vars = ['OSIM', "LogFreq", "LogCC", "G_O"]
    exog_vars = ['OSIM', "LogFreq", "LogCC", "G_J"]
    exog_vars = ['OSIM', "LogFreq", "LogCC", "G_Co"]
    exog_vars = ['OSIM', "LogFreq", "LogCC", "G_A", "G_O"]
    exog_vars = ['OSIM', "LogFreq", "LogCC", "G_A", "G_O", "G_J"]
    exog_vars = ['OSIM', "LogFreq", "LogCC", "G_A", "G_O", "G_J","G_Co"]
    
    exog = sm.add_constant(data[exog_vars])
    mod = PanelOLS(data.SSIM, exog, entity_effects=True, time_effects=True)
    fe_re_res = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    fe_re_pred = fe_re_res.predict()
    print(fe_re_res)
    
    #  Table 5
    # 固定效应 - 双向效应
    exog_vars = ['ODIS']
    exog_vars = ['ODIS', "LogFreq"]
    exog_vars = ['ODIS', "LogFreq", "LogCC"]
    exog_vars = ['ODIS', "LogFreq", "LogCC", "G_A"]
    exog_vars = ['ODIS', "LogFreq", "LogCC", "G_O"]
    exog_vars = ['ODIS', "LogFreq", "LogCC", "G_J"]
    exog_vars = ['ODIS', "LogFreq", "LogCC", "G_Co"]
    exog_vars = ['ODIS', "LogFreq", "LogCC", "G_A", "G_O"]
    exog_vars = ['ODIS', "LogFreq", "LogCC", "G_A", "G_O", "G_J"]
    exog_vars = ['ODIS', "LogFreq", "LogCC", "G_A", "G_O", "G_J","G_Co"]
    
    exog = sm.add_constant(data[exog_vars])
    mod = PanelOLS(data["SDIS"], exog, entity_effects=True, time_effects=True)
    fe_re_res = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    fe_re_pred = fe_re_res.predict()
    print(fe_re_res)
    
    # 绘图
    plt_X = list()
    plt_Y_actual = list()
    plt_Y_pred   = list()
    for fos in tqdm(Examples):
        plt_x = data.loc[fos]['OSIM']
        plt_y_acutal = data.loc[fos]['SSIM']
        plt_y_pred   = fe_re_pred.loc[fos]['fitted_values']
        plt_X.append(plt_x)
        plt_Y_actual.append(plt_y_acutal)
        plt_Y_pred.append(plt_y_pred)
    plt_X = np.concatenate(plt_X)
    plt_Y_actual = np.concatenate(plt_Y_actual)
    plt_Y_pred   = np.concatenate(plt_Y_pred)
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman", # 
              "font.size" : 20
              }
    rcParams.update(config)
    # 散点图
    plt.scatter(plt_X, plt_Y_actual, c='gray', alpha=0.1, s=1)
    plt.scatter(plt_X, plt_Y_pred,   c='red',  alpha=0.1, s=1, marker="o")
    plt.ylabel("$SSIM$")
    plt.xlabel("$OSIM$")
    plt.yticks(np.arange(0.1, 0.6, 0.1))
    plt.xticks(np.arange(-0.3, 0.3, 0.1))
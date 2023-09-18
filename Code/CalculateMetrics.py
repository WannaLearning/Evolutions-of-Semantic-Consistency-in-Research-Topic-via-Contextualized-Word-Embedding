#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:11:40 2023

@author: aixuexi
"""
import math
import pickle
import random
import multiprocessing
import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer
from sklearn import decomposition


# Metrics for indentifying the evolution pattern of topic embeddings:
# Self-similarity, MEV(PCA), Self-distance

def get_vec_func(vec_dict, beg_year, end_year):
    """返回beg_year至end_year的fid_1的向量组"""
    year_list = vec_dict.keys()
    year_list = sorted(year_list)
    vec_arr = list()
    for year in year_list:
        if beg_year <= year and year <= end_year:
            vec_year = vec_dict[year]
            for vec in vec_year:
                vec_arr.append([vec])
    if len(vec_arr) > 0:
        vec_arr = np.concatenate(vec_arr, axis=0)
    return vec_arr


#%%
# computer cosine similarity among topic embeddings 

def calculate_self_similarity_func(vec_1, vec_2, bSameFoS, sampling_size, max_mp_num, 
                                   min_sampling_times, max_sampling_times, sampling_discount_ratio):
    num_of_vecs_1  = len(vec_1)
    num_of_vecs_2  = len(vec_2)
    sampling_times = math.ceil(num_of_vecs_1 / sampling_size)  # 所有进程的累计采样次数
    
    if sampling_times > min_sampling_times:
        sampling_times = int(sampling_times * sampling_discount_ratio)
        sampling_times = min(sampling_times,  max_sampling_times)
    
    # 确定所需进程数目
    if sampling_times <= max_mp_num:
        mp_num = sampling_times
    else:
        mp_num = max_mp_num
    
    # 需要采样 + 多进程, 取平均值   (num_of_vecs_1 > sampling_size)
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        # 每个进程计算 sampling_times_i 次采样过程
        sampling_times_i = int(sampling_times / mp_num)
        # print(mp_i)
        vec_i_1 = list()
        vec_i_2 = list()
        for sampling_j in range(sampling_times_i):
            # sampling from vec_1
            sampling_index_1 = np.arange(num_of_vecs_1)
            random.shuffle(sampling_index_1)
            sampling_index_j_1 = sampling_index_1[:sampling_size]
            vec_j_1 = vec_1[sampling_index_j_1, :]
            vec_i_1.append(vec_j_1)
            
            if not bSameFoS:
                # sampling from vec_2
                sampling_index_2 = np.arange(num_of_vecs_2)
                random.shuffle(sampling_index_2)
                sampling_index_j_2 = sampling_index_2[:sampling_size]
                vec_j_2 = vec_2[sampling_index_j_2, :]
                vec_i_2.append(vec_j_2) 
            
        if bSameFoS:
            results.append(pool.apply_async(calculate_self_similarity_loop, (vec_i_1, vec_i_1, True,))) 
        else:
            results.append(pool.apply_async(calculate_self_similarity_loop, (vec_i_1, vec_i_2, False,)))
    
    pool.close()
    pool.join()
    
    self_cos = list()
    for res in results:
        self_cos_i = res.get()
        self_cos.append(self_cos_i)
    self_cos = np.mean(self_cos)  # 多个进程的自相似性得分再平均
        
    return self_cos


def calculate_self_similarity_loop(vec_i_1, vec_i_2, bSameFoS):
    """采样计算self-similarity"""
    if bSameFoS:
        self_cos_i = list()
        for vec_j in vec_i_1:
            self_cos_j = calculate_self_similarity(vec_j, vec_j, bSameFoS)
            self_cos_i.append(self_cos_j)
    else:
        self_cos_i = list()
        for vec_j_1, vec_j_2 in zip(vec_i_1, vec_i_2):
            self_cos_j = calculate_self_similarity(vec_j_1, vec_j_2, bSameFoS)
            self_cos_i.append(self_cos_j)
    return np.mean(self_cos_i)


def calculate_self_similarity(vec_1, vec_2, bSameFoS, max_dim=3000, eps=1e-3):
    """self-similarity(vec_1, vec_2)"""  
    # num_1 或 num_2 维度太大, cosine_similarity爆内存
    cos_sum = 0
    num_1, _ = vec_1.shape
    num_2, _ = vec_2.shape
    split_1 = math.ceil(num_1 / max_dim)
    split_2 = math.ceil(num_2 / max_dim)
    beg_1, end_1 = 0, 0
    for i in range(split_1):
        end_1 = min((i + 1) * max_dim, num_1)
        beg_2, end_2 = 0, 0
        for j in range(split_2):
            end_2 = min((j + 1) * max_dim, num_2)
            # print("row: {}-{}".format(beg_1, end_1), "col: {}-{}".format(beg_2, end_2))
            vec_1_i = vec_1[beg_1: end_1]
            vec_2_j = vec_2[beg_2: end_2]
            cos_ij = Cosine_distance(vec_1_i, vec_2_j)
            cos_ij_sum = np.sum(cos_ij / num_1)  # 提前除, 避免内存不足
            cos_sum += cos_ij_sum
            
            beg_2 = end_2 
        beg_1 = end_1
    
    if bSameFoS: # 同一个FoS需要剔除对角线处Cosine==1
        self_cos = (cos_sum - 1) / (1 * (num_1-1) + eps)
    else:
        self_cos = cos_sum / (1 * num_2)
    return self_cos


def Cosine_distance(matrix1, matrix2):
    """计算两个矩阵间所有行向量间cosine"""
    vec_1_i = torch.tensor(matrix1).float()
    vec_2_j = torch.tensor(matrix2).float()
    cos_ij  = torch.cosine_similarity(torch.unsqueeze(vec_1_i, dim=1), 
                                      torch.unsqueeze(vec_2_j, dim=0), dim=-1, eps=1e-5)
    cosine_distance  = np.array(cos_ij.numpy(), dtype=np.float16)
    return cosine_distance

#%%
# calculate Euclidean distance among topic embeddings

def calculate_self_distance_func(vec_1, vec_2, bSameFoS, sampling_size, max_mp_num, 
                                 min_sampling_times, max_sampling_times, sampling_discount_ratio):
                                   
    num_of_vecs_1  = len(vec_1)
    num_of_vecs_2  = len(vec_2)
    sampling_times = math.ceil(num_of_vecs_1 / sampling_size)  # 所有进程的累计采样次数
    
    if sampling_times > min_sampling_times:
        sampling_times = int(sampling_times * sampling_discount_ratio)
        sampling_times = min(sampling_times,  max_sampling_times)
    
    if sampling_times <= max_mp_num:
        mp_num = sampling_times
    else:
        mp_num = max_mp_num
        
    results = list()
    pool = multiprocessing.Pool(processes=mp_num)      # 创建进程池
    for mp_i in range(mp_num):
        # 每个进程计算 sampling_times_i 次采样过程
        sampling_times_i = int(sampling_times / mp_num)
        # print(mp_i)
        vec_i_1 = list()
        vec_i_2 = list()
        for sampling_j in range(sampling_times_i):
            # sampling from vec_1
            sampling_index_1 = np.arange(num_of_vecs_1)
            random.shuffle(sampling_index_1)
            sampling_index_j_1 = sampling_index_1[:sampling_size]
            vec_j_1 = vec_1[sampling_index_j_1, :]
            vec_i_1.append(vec_j_1)
            
            if not bSameFoS:
                # sampling from vec_2
                sampling_index_2 = np.arange(num_of_vecs_2)
                random.shuffle(sampling_index_2)
                sampling_index_j_2 = sampling_index_2[:sampling_size]
                vec_j_2 = vec_2[sampling_index_j_2, :]
                vec_i_2.append(vec_j_2)
            
        if bSameFoS:
            results.append(pool.apply_async(calculate_self_distance_loop, (vec_i_1, vec_i_1, True,)))
        else:
            results.append(pool.apply_async(calculate_self_distance_loop, (vec_i_1, vec_i_2, False,)))
                                        
    pool.close()
    pool.join()
    
    self_dis = list()
    for res in results:
        self_dis_i = res.get()
        self_dis.append(self_dis_i)
    self_dis = np.mean(self_dis)  # 多个进程的自距离得分再平均
 
    return self_dis


def calculate_self_distance_loop(vec_i_1, vec_i_2, bSameFoS):
    """采样计算self-distance"""
    if bSameFoS:
        self_dis_i = list()
        for vec_j in vec_i_1:
            self_dis_j = calculate_self_distance(vec_j, vec_j, bSameFoS)
            self_dis_i.append(self_dis_j)
    else:
        self_dis_i = list()
        for vec_j_1, vec_j_2 in zip(vec_i_1, vec_i_2):
            self_dis_j = calculate_self_distance(vec_j_1, vec_j_2, bSameFoS)
            self_dis_i.append(self_dis_j)
    return np.mean(self_dis_i)


def calculate_self_distance(vec_1, vec_2, bSameFoS, max_dim=4000, eps=1e-3):
    """self-distance(vec_1, vec_2)"""  
    vec_1 = torch.tensor(vec_1).float()
    vec_2 = torch.tensor(vec_2).float()
    num_1, _ = vec_1.shape
    num_2, _ = vec_2.shape
    
    # num_1 或 num_2 维度太大, cosine_similarity 内存不足
    dis_sum = 0
    split_1 = math.ceil(num_1 / max_dim)
    split_2 = math.ceil(num_2 / max_dim)
    beg_1, end_1 = 0, 0
    for i in range(split_1):
        end_1 = min((i + 1) * max_dim, num_1)
        beg_2, end_2 = 0, 0
        for j in range(split_2):
            end_2 = min((j + 1) * max_dim, num_2)
            # print("row: {}-{}".format(beg_1, end_1), "col: {}-{}".format(beg_2, end_2))
            vec_1_i = vec_1[beg_1: end_1]
            vec_2_j = vec_2[beg_2: end_2]
            dis_ij  = L2_distance(vec_1_i, vec_2_j)
            # torch.dist(vec_1[0, :], vec_2[-2, :])
            dis_ij  = np.array(dis_ij.numpy(), dtype=np.float16)
            dis_ij_sum = np.sum(dis_ij / num_1)
            dis_sum += dis_ij_sum
            beg_2 = end_2 
        beg_1 = end_1
    
    if bSameFoS:  # 剔除对角线距离为0
        self_dis = dis_sum / (1 * (num_1-1) + eps)
    else:
        self_dis = dis_sum / (1 * num_2)
    return self_dis


def L2_distance(a, b):
    # 使用矩阵运算的方式求取两个矩阵(a和b)中各个样本的欧式距离
    m = a.shape[0]
    n = b.shape[0]
    # 对矩阵的每个元素求平方
    aa = torch.pow(a, 2)   # [m, d]
    # 按行求和, 并且保留维度数量不变
    aa = torch.sum(aa, dim=1, keepdim=True)   # [m, 1]
    # 将矩阵aa从[m, 1]的形状扩展为[m, n]
    aa = aa.expand(m, n)   # [m, n]
    # 处理矩阵b
    bb = torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n ,m)   # [n, m]
    bb = torch.transpose(bb, 0, 1)   # [m, n]
    # 计算第三项   [m, d] * [d, n] = [m, n]
    tail = 2 * torch.matmul(a, torch.transpose(b, 0, 1))
    distance = torch.sqrt(torch.maximum(aa + bb - tail, torch.tensor(0.0)))
    return distance


#%%
# computer maximum explainable variance in PCA

def maximum_explainable_variance_func(vec_1, n_components=1):
    """PCA - maximum_explainable_variance"""
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(vec_1.transpose())
    # vec_1_ = pca.transform(vec_1.transpose())
    mer = pca.explained_variance_ratio_[0]
    return mer

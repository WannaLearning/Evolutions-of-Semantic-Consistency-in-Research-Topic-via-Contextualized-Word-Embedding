#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:59:31 2023

@author: aixuexi
"""
import os
import re
import json
import pickle
import prettytable as pt
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from matplotlib import rcParams

from Utils import save_file, read_file, abs_file_path


#%%
# OAG2.1 can be downloaded from https://www.aminer.cn/oag-2-1
# extract papers in the computer science field from OAG dataset
# the code can be found in ExtractCSPapers.py


#%%
# FieldsOfStudy info
# extract fos level information in MAG

def extract_fos_hierachical_relationship_info():
    """ Read hierarchical relationship (FoS0, FoS1, ..., FoS6) from FieldsOfStudy.nt and FieldOfStudyChildren.nt """
    Fos_file_1 = "./temp/FieldsOfStudy.nt"
    Fos_file_2 = "./temp/FieldOfStudyChildren.nt"
    
    # process xml
    tmp1 = list()
    tmp2 = list()
    tmp3 = list()
    with open(Fos_file_1, 'r') as f:
        while True:
            oneline = f.readline().strip()
            if oneline:
                oneline_split = oneline.split(" ")
                if oneline_split[1] == "<http://xmlns.com/foaf/0.1/name>":    
                    fid  = re.findall("\d+", oneline)[0]
                    name = re.findall(r'"([^"]+)"', oneline)[0]
                    tmp1.append(fid)
                    tmp2.append(name)
                    # print(oneline, name, fid)
                if oneline_split[1] == "<http://ma-graph.org/property/level>":            
                    level = re.findall(r'"([^"]+)"', oneline)[0]
                    tmp3.append(level)
                    # print(oneline, level)
            else:
                break
    # field of study - level and name
    FoSs_info = dict()
    for fid, name, level in zip(tmp1, tmp2, tmp3):
        FoSs_info[fid] = (name, int(level))
    
    # field of study - hierarchical relationship
    FoSs_childs = list()
    with open(Fos_file_2, 'r') as f:
        while True:
            oneline = f.readline().strip()
            if oneline:
                oneline_split = oneline.split(" ")
                child  = oneline_split[0]
                parent = oneline_split[2]
                child_id  = re.findall("\d+", child)[0]
                parent_id = re.findall("\d+", parent)[0]
                FoSs_childs.append((child_id, parent_id))
            else:
                break
    FoSs_childs_ = dict()
    for child, parent in FoSs_childs:
        if parent not in FoSs_childs_:
            FoSs_childs_[parent] = list()
        FoSs_childs_[parent].append(child)
    FoSs_childs  = FoSs_childs_
    save_file(FoSs_info,   "./temp/FoSs_info.pkl")
    save_file(FoSs_childs, "./temp/FoSs_childs.pkl")
    
    
def select_fos_level_from_field(level0='Computer science', selected_level=[2]):
    """ select level_i FoS from a specific field, level0 """

    FoSs_info   = read_file("./temp/FoSs_info.pkl")
    FoSs_childs = read_file("./temp/FoSs_childs.pkl")

    level2name = dict()
    name2fid   = dict()
    for fid in FoSs_info:
        name, level = FoSs_info[fid]
        name2fid[name] = fid
        if level not in level2name:
            level2name[level] = list()
        level2name[level].append(name)
    
    def find_childs(fid, FoSs_childs):
        # 递归寻找fid下所有的下级fos
        all_childs_fid  = list()
        if fid not in FoSs_childs:  # 该fid无下级fid
            return all_childs_fid
        else:
            childs_fid      = FoSs_childs[fid]
            all_childs_fid += childs_fid
            for fid_i in childs_fid:
                childs_fid_i    = find_childs(fid_i, FoSs_childs)
                all_childs_fid += childs_fid_i
            return all_childs_fid
    
    # 挑选特定领域level0下FoS
    level0_id = name2fid[level0]
    all_childs_fid = find_childs(level0_id, FoSs_childs)
    all_childs_fid = {fid: "" for fid in all_childs_fid}
    
    level2name_cs = dict()
    for fid in FoSs_info:
        if fid not in all_childs_fid:
            continue
        name, level = FoSs_info[fid]
        if level not in level2name_cs:
            level2name_cs[level] = list()
        level2name_cs[level].append(name)
    
    tb = pt.PrettyTable()
    tb.title = level0 + "({})".format(selected_level)
    tb.field_names = ["MAG-LEVEL", "Total Num (FoS)", "Num in the field (FoS)"]
    for level in range(0, 6):
        if level == 0:
            tb.add_row([level, len(level2name[level]), 1])
        else:
            tb.add_row([level, len(level2name[level]), len(level2name_cs[level])])
    print(tb)
     
    # 挑选特定领域level0下特定level的FoS
    filtered_FoSs = dict()
    for level in selected_level:
        for fos in level2name_cs[level]:
            filtered_FoSs[fos.lower()] = ""
    return  filtered_FoSs


#%%
def extract_fos_related_info():
    """ 
    statistics of FoS: adoption frequency, citation frequency, author num, orgnization num, journal/conference num
    采纳频次, 被引用频次, 作者数目, 机构数目, 期刊数目 / 逐年
    """
    # 创建文件夹, 其中包括FoS的逐年相关信息
    save_path = os.path.join(abs_file_path, "FoS2Info")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # read fos level_i information
    if os.path.exists("./temp/FoSs_info.pkl") and os.path.exists("./temp/FoSs_childs.pkl"):
        filtered_FoSs = select_fos_level_from_field(level0='Computer science', selected_level=[2])
    else:
        extract_fos_hierachical_relationship_info()
        filtered_FoSs = select_fos_level_from_field(level0='Computer science', selected_level=[2])
    
    # read citaion/ adoption frequency
    # file_fos_i is obtained by executing ExtractCSPapers.py 
    file_fos_i = "/mnt/disk2/MAG_DATA_SET/MAGv2.1-meta-computer science/mag_papers_{}.txt"  
    for i in tqdm(range(0, 17)):
        FoS2Aid = dict()  # FoS逐年被Aid采纳*次
        FoS2Oid = dict()  # FoS逐年被Oid采纳*次
        FoS2Vid = dict()  # FoS逐年被Vid采纳*次
        FoS2Fid = dict()  # FoS逐年与Fid共同出现*次 (和自身出现即采纳频率)
        Pid2CC  = dict()  # Pid逐年的被引用次数 (计算FoS的引用频率)
        with open(file_fos_i.format(i), 'r') as f:
            while True:
                oneline = f.readline().strip()
                if oneline:
                    oneline_json = json.loads(oneline)
                    FoSs = oneline_json['f']
                    FoSs_ = list()
                    for fos in FoSs:
                        if fos in filtered_FoSs:  # 在select_fos_level挑选的FoS中
                            FoSs_.append(fos)
                    if len(FoSs_) == 0:
                        continue
                    
                    time = oneline_json['t']
                    aids_oids = oneline_json['aid']
                    aids = [aid for aid, oid in aids_oids]  # 作者 ID
                    oids = [oid for aid, oid in aids_oids]  # 机构 ID (会重复)
                    vid  = oneline_json['v']                # 期刊 ID
                    cited_pids = oneline_json['r']          # 参考文献 ID
                    for fos in FoSs_: # Level 2 FoS
                        if fos not in FoS2Aid:
                            FoS2Aid[fos] = dict()
                        if time not in FoS2Aid[fos]:
                            FoS2Aid[fos][time] = dict()
                        for aid in aids:
                            if aid not in FoS2Aid[fos][time]:
                                FoS2Aid[fos][time][aid] = 0
                            FoS2Aid[fos][time][aid] += 1    # aid在time时采用fos1次
                        
                        if fos not in FoS2Oid:
                            FoS2Oid[fos] = dict()
                        if time not in FoS2Oid[fos]:
                            FoS2Oid[fos][time] = dict()
                        for oid in oids:
                            if oid not in FoS2Oid[fos][time]:
                                FoS2Oid[fos][time][oid] = 0
                            FoS2Oid[fos][time][oid] += 1    # oid在time时采用fos1次
                        
                        if fos not in FoS2Vid:
                            FoS2Vid[fos] = dict()
                        if time not in FoS2Vid[fos]:
                            FoS2Vid[fos][time] = dict()
                        if vid not in FoS2Vid[fos][time]:
                            FoS2Vid[fos][time][vid] = 0
                        FoS2Vid[fos][time][vid] += 1        # vid在time时采用fos1次
                        
                        if fos not in FoS2Fid:
                            FoS2Fid[fos] = dict()
                        if time not in FoS2Fid[fos]:
                            FoS2Fid[fos][time] = dict()
                        for fos_j in FoSs_:
                            if fos_j not in FoS2Fid[fos][time]:
                                FoS2Fid[fos][time][fos_j] = 0
                            FoS2Fid[fos][time][fos_j] += 1  # fos与fos_j在time时共现1次
                    
                    for pid_ in cited_pids:
                        if pid_ not in Pid2CC:
                            Pid2CC[pid_] = dict()
                        if time not in Pid2CC[pid_]:
                            Pid2CC[pid_][time] = 0
                        Pid2CC[pid_][time] += 1             # pid_在time时被引用1次  
                else:
                    break
        # save
        save_file(FoS2Aid, os.path.join(save_path, "FoS2Aid_{}.pkl".format(i)))
        save_file(FoS2Oid, os.path.join(save_path, "FoS2Oid_{}.pkl".format(i)))
        save_file(FoS2Vid, os.path.join(save_path, "FoS2Vid_{}.pkl".format(i)))
        save_file(FoS2Fid, os.path.join(save_path, "FoS2Fid_{}.pkl".format(i)))
        save_file(Pid2CC,  os.path.join(save_path, "Pid2CC_{}.pkl".format(i)))
    
    # (1) 合并信息 - 作者数目, 机构数目, 期刊数目, FoS共现数目
    if not os.path.exists(os.path.join(save_path, "FoS2Aid")):
        os.mkdir(os.path.join(save_path, "FoS2Aid"))
    if not os.path.exists(os.path.join(save_path, "FoS2Oid")):
        os.mkdir(os.path.join(save_path, "FoS2Oid"))
    if not os.path.exists(os.path.join(save_path, "FoS2Vid")):
        os.mkdir(os.path.join(save_path, "FoS2Vid"))
    if not os.path.exists(os.path.join(save_path, "FoS2Fid")):
        os.mkdir(os.path.join(save_path, "FoS2Fid"))

    def merge_func(FoS2Aid, fos, dir_name):
        """合并函数"""
        fos_path     = os.path.join(save_path, "{}/{}.pkl".format(dir_name, fos))
        current_file = FoS2Aid[fos]
        if not os.path.exists(fos_path):
            save_file(current_file, fos_path)
        else:
            former_file = read_file(fos_path) 
            for year in current_file:
                if year not in former_file:
                    former_file[year] = current_file[year]
                else:
                    for aid in current_file[year]:
                        if aid not in former_file[year]:
                            former_file[year][aid] = current_file[year][aid]
                        else:
                            former_file[year][aid] += current_file[year][aid]
            save_file(former_file, fos_path)
    
    for i in tqdm(range(0, 17)):
        FoS2Aid  = read_file(os.path.join(save_path, "FoS2Aid_{}.pkl".format(i)))
        for fos in FoS2Aid:
            merge_func(FoS2Aid, fos, "FoS2Aid")
        FoS2Oid = read_file(os.path.join(save_path, "FoS2Oid_{}.pkl".format(i)))
        for fos in FoS2Oid:
            merge_func(FoS2Oid, fos, "FoS2Oid")
        FoS2Vid = read_file(os.path.join(save_path, "FoS2Vid_{}.pkl".format(i)))
        for fos in FoS2Vid:
            merge_func(FoS2Vid, fos, "FoS2Vid")
        FoS2Fid = read_file(os.path.join(save_path, "FoS2Fid_{}.pkl".format(i)))
        for fos in FoS2Fid:
            merge_func(FoS2Fid, fos, "FoS2Fid")
    

    # (2) 统计引用数目
    if not os.path.exists(os.path.join(save_path, "FoS2CC")):
        os.mkdir(os.path.join(save_path, "FoS2CC"))
    
    # 将Pid2CC_i融合 - 即论文的被引用频次
    Pid2CC = dict()
    for i in tqdm(range(0, 17)):
        Pid2CC_i = read_file(os.path.join(save_path, "Pid2CC_{}.pkl".format(i)))
        for pid in Pid2CC_i:
            if pid not in Pid2CC:
                Pid2CC[pid] = Pid2CC_i[pid]
            else:
                for year in Pid2CC_i[pid]:
                    if year not in Pid2CC[pid]:
                        Pid2CC[pid][year] = Pid2CC_i[pid][year]
                    else:
                        Pid2CC[pid][year] += Pid2CC_i[pid][year]
                 
    for i in tqdm(range(0, 17)):
        FoS2CC = dict()
        with open(file_fos_i.format(i), 'r') as f:
            while True:
                oneline = f.readline().strip()
                if oneline:
                    oneline_json = json.loads(oneline)
                    FoSs = oneline_json['f']
                    FoSs_ = list()
                    for fos in FoSs:
                        if fos in filtered_FoSs:  # 在select_fos_level挑选的FoS中
                            FoSs_.append(fos)
                    if len(FoSs_) == 0:
                        continue
                    
                    pid  = oneline_json['pid']
                    time = oneline_json['t']
                    if pid not in Pid2CC:         # 该论文未被引用过
                        continue
                    
                    for fos in FoSs_:
                        if fos not in FoS2CC:
                            FoS2CC[fos] = dict()
                        for citing_year in Pid2CC[pid]:
                            if citing_year not in FoS2CC[fos]:
                                FoS2CC[fos][citing_year] = np.zeros(2)
                            cc_full_num = Pid2CC[pid][citing_year]
                            cc_frac_num = Pid2CC[pid][citing_year] / len(FoSs_)
                            FoS2CC[fos][citing_year][0] = cc_full_num
                            FoS2CC[fos][citing_year][1] = cc_frac_num
                else:
                    break
        save_file(FoS2CC, os.path.join(save_path, "FoS2CC_{}.pkl".format(i)))
    
    # 合并信息 - FoS的被引用频次
    for i in tqdm(range(0, 17)):
        FoS2CC = read_file(os.path.join(save_path, "FoS2CC_{}.pkl".format(i)))
        for fos in FoS2CC:
            fos_path = os.path.join(save_path, "{}/{}.pkl".format("FoS2CC", fos))    
            current_file = FoS2CC[fos]
            if not os.path.exists(fos_path):
                save_file(current_file, fos_path)
            else:
                former_file = read_file(fos_path) 
                for year in current_file:
                    if year not in former_file:
                        former_file[year] = current_file[year]
                    else:
                        former_file[year] += current_file[year]
                save_file(former_file, fos_path)

    # 删除中间信息
    for i in tqdm(range(0, 17)):
        os.remove(os.path.join(save_path, "FoS2Aid_{}.pkl".format(i)))
        os.remove(os.path.join(save_path, "FoS2Oid_{}.pkl".format(i)))
        os.remove(os.path.join(save_path, "FoS2Vid_{}.pkl".format(i)))
        os.remove(os.path.join(save_path, "FoS2Fid_{}.pkl".format(i)))
        os.remove(os.path.join(save_path, "Pid2CC_{}.pkl".format(i)))
        os.remove(os.path.join(save_path, "FoS2CC_{}.pkl".format(i)))

        
def plot_statistics_of_dataset_employed_in_our_paper():
    # plot Fig.2 and Fig.3
    
    filtered_FoSs = select_fos_level_from_field(level0='Computer science', selected_level=[2])
    FoSs2NoP = dict()
    for fos in filtered_FoSs:                                        
        FoSs2NoP[fos] = dict()
    
    # file_fos_i is obtained by executing ExtractCSPapers.py 
    file_fos_i = "/mnt/disk2/MAG_DATA_SET/MAGv2.1-meta-computer science/mag_papers_{}.txt"
    yearly_papers   = dict()  # 逐年发文量
    yearly_papers_2 = dict()  # 涵盖FoS_L2的发文量
    for i in tqdm(range(0, 17)):
        with open(file_fos_i.format(i), 'r') as f:
            while True:
                oneline = f.readline().strip()
                if oneline:
                    oneline_json = json.loads(oneline)
                    FoSs = oneline_json['f']
                    year = oneline_json['t']
                    # 统计 papers in CS
                    if year not in yearly_papers:
                        yearly_papers[year] = 1
                    else:
                        yearly_papers[year] += 1
                    # 在select_fos_level挑选的FoS集合
                    FoSs_ = list()
                    for fos in FoSs:
                        if fos in filtered_FoSs:  
                            FoSs_.append(fos)
                            if year not in FoSs2NoP[fos]:
                                FoSs2NoP[fos][year] = 1
                            else:
                                FoSs2NoP[fos][year] += 1  # 统计 FoS2 下论文数
                    if len(FoSs_) == 0:
                        continue
                    # 统计 papers that studys L2 in CS
                    if year not in yearly_papers_2:
                        yearly_papers_2[year] = 1
                    else:
                        yearly_papers_2[year] += 1
                else:
                    break
                
    # Fig.2 annual publications in the computer science field
    X = np.arange(1900, 2020)
    Y = list()
    Y2= list()
    for x in X:
        if x in yearly_papers:
            Y.append(yearly_papers[x])
        else:
            Y.append(0)
        if x in yearly_papers_2:
            Y2.append(yearly_papers_2[x])
        else:
            Y2.append(0)
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    plt.plot(X, Y, c='blue', linewidth=1.5, label='Papers in the CS', linestyle='--')
    plt.plot(X, Y2, c='black', linewidth=1.5, label=r"Papers with at least one $FoS_{L2}$ in the CS")
    plt.yscale("log")
    plt.xticks(np.arange(1800, 2025, 10), rotation=45)
    plt.xlabel("Time")
    plt.ylabel("Annual number of publications")
    
    cut_x = [1990, 2018]
    cut_y = [yearly_papers_2[1990], yearly_papers_2[2018]]
    plt.legend(frameon=False, fontsize=18)
    plt.xlim(1900, 2020)   
           
    # Fig. 3 adoption frequency of FoS
    FoS2NoPSum = dict()
    nop2not    = dict()
    for fos in FoSs2NoP:
        NoPSum = 0 
        for year in np.arange(1990, 2019):
            if year in FoSs2NoP[fos]:
                NoPSum += FoSs2NoP[fos][year]
        FoS2NoPSum[fos] = NoPSum  
    for fos in FoS2NoPSum:
        NoPSum = FoS2NoPSum[fos]
        if NoPSum not in nop2not:
            nop2not[NoPSum] = 1
        else:
            nop2not[NoPSum] += 1
    
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 20
              }
    rcParams.update(config)
    
    X = sorted(list(nop2not.keys())) 
    Y = [nop2not[x] for x in X]
    plt.scatter(X, Y, linewidth=.5, c='black', s=10, marker="+")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Number of papers studying a topic")
    plt.ylabel(r"Number of topics $(FoS_{L2})$")
    
    cut_x = [100, max(X)]
    cut_y = [Y[100], 1]
    plt.fill_between(cut_x, [0, 0], cut_y, alpha=0.15, color='gray')
    plt.text(0.75, 0.3, r"7,132 topics", 
             transform=plt.gca().transAxes, color='black', fontsize=18) 
    plt.text(0.75, 0.15, "{}-{}\n".format("1990", "2018"), 
             transform=plt.gca().transAxes, color='black', fontsize=18)  


def main():
    extract_fos_hierachical_relationship_info()
    extract_fos_related_info()
    plot_statistics_of_dataset_employed_in_our_paper()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:04:26 2023

@author: aixuexi
"""
import os
import json
import pickle
import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer
from tqdm import tqdm

from Utils import save_file, read_file, abs_file_path
from DataCollectionAndPreprocess import select_fos_level_from_field


# create Contextualized Embedding of [FoS] by the BERT-based model

def GenerateTopicEmbeddings():
    """
    BERT Model:
    Input: [CLS] + FoS + [SEP] + Title + Abstract
    Output: The average of FoS vectors
    """
    
    filtered_FoSs = select_fos_level_from_field(level0='Computer science', selected_level=[2])
    
    pretrained_path = os.path.join(os.getcwd(), 'bert-base-uncased') 
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = BertModel.from_pretrained(pretrained_path)
    for params in model.parameters():
        params.requires_grad = False
    model     = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    
    file_content_i = "/mnt/disk2/MAG_DATA_SET/MAGv2.1-abstract-cs/mag_papers_{}"
    file_fos_i     = "/mnt/disk2/MAG_DATA_SET/MAGv2.1-meta-computer science/mag_papers_{}.txt"  
      
    for i in range(0, 17):
        # read title + abstract
        pid2content = dict()
        with open(file_content_i.format(i), 'r') as f:
            while True:
                oneline = f.readline().strip()
                if oneline:
                    oneline_json = json.loads(oneline)
                    pid = oneline_json['pid']                 # 论文id
                    content  = oneline_json["content"].strip()
                    con_spl  = content.split(";")
                    year     = con_spl[0].strip()             # 出版年份
                    title    = con_spl[1].strip()             # 标题
                    abstract = ";".join(con_spl[2:]).strip()  # 摘要
                    if len(title) == 0 and len(abstract) == 0:
                        continue
                    content  = title + "." + abstract
                    pid2content[pid] = content
                else:
                    break
        # read FoSs   
        pid2FoSs = dict()
        with open(file_fos_i.format(i), 'r') as f:
            while True:
                oneline = f.readline().strip()
                if oneline:
                    oneline_json = json.loads(oneline)
                    year = oneline_json['t']
                    pid  = oneline_json['pid']
                    FoSs = oneline_json['f']
                    FoSs_ = list()
                    for fos in FoSs:
                        if fos in filtered_FoSs:  # 在select_fos_level挑选的FoS中
                            FoSs_.append(fos)
                    if len(FoSs_) > 0:
                        pid2FoSs[pid] = (FoSs_, year)
                else:
                    break
        print("({}) 论文数目: {}/{}".format(i, len(pid2FoSs), len(pid2content)))   
        
        # 嵌入BERT向量
        batch_size  = 150
        max_length  = 256     # bert model truncation length
        batchs_sen  = list()  # fos + "[SEP]" + content # content = title + "." + abstract
        batchs_fos  = list()  # fos
        batchs_year = list()  # publication year
        results     = dict()  # fos -> key -> vec 
        for c, pid in tqdm(enumerate(pid2FoSs)):
            FoSs, year = pid2FoSs[pid]
            if pid not in pid2content:
                continue
            content = pid2content[pid]
            for fos in FoSs:
                sentence = [fos, content]
                batchs_sen.append(sentence)
                batchs_fos.append(fos)
                batchs_year.append(year)
            if len(batchs_sen) >= batch_size or c == len(pid2FoSs)-1:
                # 开始2Vec
                # tokenize 输入数据
                input_for_bert = tokenizer(batchs_sen, padding='max_length', max_length=max_length, truncation=True)
                input_ids      = torch.tensor(input_for_bert['input_ids']).to(device)
                attention_mask = torch.tensor(input_for_bert['attention_mask']).to(device)
                token_type_ids = torch.tensor(input_for_bert['token_type_ids']).to(device)
                
                # 获取last_hidden_state
                with torch.no_grad():
                   outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   output_hidden_states=True, output_attentions=False) 
                last_hidden_state = outputs.last_hidden_state 
                
                # fos的vec
                Key_Vec  = torch.multiply(1 - token_type_ids, attention_mask)
                Key_Vec[:, 0] = 0 # [CLS]
                SEP_idx = torch.sum(Key_Vec, dim=-1, keepdim=True)
                Key_Vec.scatter_(1, SEP_idx, 0) # [SEP]
                Key_Vec_ = torch.unsqueeze(Key_Vec, dim=-1)
                
                Key_Sum  = torch.multiply(last_hidden_state, Key_Vec_)
                Key_Sum  = torch.sum(Key_Sum, dim=1)
                Key_div  = torch.sum(Key_Vec, dim=-1, keepdim=True)
                Key_Avg  = torch.divide(Key_Sum, Key_div)  # 关键词的平均表征 (batch_size x 768)
                batchs_vec = Key_Avg.cpu().numpy()
                
                # 存放结果
                for j in range(len(batchs_fos)):
                    fos  = batchs_fos[j]
                    year = batchs_year[j]
                    vec  = batchs_vec[j]
                    if fos not in results:
                        results[fos] = dict()
                    if year not in results[fos]:
                        results[fos][year] = list()
                    results[fos][year].append(np.array(vec, dtype=(np.float16)))
                
                # 清空缓存
                batchs_sen  = list()
                batchs_fos  = list()
                batchs_year = list()
                batchs_vec  = list()
                torch.cuda.empty_cache()
        
        del pid2content, pid2FoSs
        # 储存结果
        with open("/mnt/disk2/MAG_DATA_SET/BERT-cs/vec_{}.pkl".format(i), 'wb') as f:
            pickle.dump(results, f)
        del results
        

def SaveEmbeddingsSeperately():
    """
    save vectors of each FoS seperately
    """
    # 创建文件夹, 其中存在每个FoS的逐年向量信息
    if not os.path.exists(os.path.join(abs_file_path, "FoS2Vec")):
        os.mkdir(os.path.join(abs_file_path, "FoS2Vec"))
        
    for i in range(0, 17):
        # 第 i 块的向量信息
        vecs = read_file(os.path.join(abs_file_path, "vec_{}.pkl".format(i)))
        # 分词存放
        for fid in tqdm(vecs):
            current_fos2vec = vecs[fid]
            fos_path = os.path.join(abs_file_path, "FoS2Vec")
            fos_path = os.path.join(fos_path, "{}.pkl".format(fid))
            if not os.path.exists(fos_path):
                save_file(current_fos2vec, fos_path)
            else:
                former_fos2vec = read_file(fos_path)
                for year in current_fos2vec:
                    if year not in former_fos2vec:
                        former_fos2vec[year] = current_fos2vec[year]
                    else:
                        former_fos2vec[year] += current_fos2vec[year]
                save_file(former_fos2vec, fos_path)   
        del vecs


def main():
    # generate topic embeddings based on the BERT model
    GenerateTopicEmbeddings()
    # save the topic embeddings sperately for each FoS
    SaveEmbeddingsSeperately()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:27:37 2023

@author: aixuexi
"""
import os
import re
import json
import pickle
import zipfile
import multiprocessing
from tqdm import tqdm


FilePath = "D:/Data" 
MAG      = os.path.join(FilePath, "MAGv2.1")
MAG_Meta = os.path.join(FilePath, "MAGv2.1-meta")
MAG_FoS  = os.path.join(FilePath, "MAG-fos")


def unzip_file(zip_src, dst_dir):
    '''
    解压Zip文件
    '''
    r = zipfile.is_zipfile(zip_src)
    if r:   
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)     
    else:
        print('This is not zip')


def ExtractMeta():
    """ exclude abstract field from the MAG dataset;
        extract meta info from MAGv2.1
    """
    # OAG_V2.1 中的 MAG, 剔除了abstract
    # MAGv2.1 中抽取 MAGv2.1-meta
    
    MAG_zips = os.listdir(MAG)
    MAG_zips.remove("mag_affiliations.zip")   # MAG 机构信息
    MAG_zips.remove("mag_venues.zip")         # MAG 期刊信息
    MAG_zips.remove("mag_authors_0.zip")      # MAG 作者信息
    MAG_zips.remove("mag_authors_1.zip")      # MAG 作者信息
    
    for file_zip_name in MAG_zips:
    
        file_zip_path = os.path.join(MAG, file_zip_name)
        file_id       = re.sub("\D", "", file_zip_name)
        print(file_zip_path, file_id)
        
        # 解压 zip 文件
        dst_dir = os.path.join(FilePath, "temp")
        unzip_file(file_zip_path, dst_dir)
        
        # 抽取 Meta 信息 - 写入 file_out_path
        file_out_path = os.path.join(MAG_Meta, "mag_papers_{}.txt".format(file_id))
        f_output      = open(file_out_path, 'w', encoding='utf8')
        
        for file_name in os.listdir(dst_dir):
            file_in_path = os.path.join(dst_dir, file_name)

            print(file_zip_path, file_name)            
            with open(file_in_path, 'r', encoding='utf8') as f:
                while True:
                    oneline = f.readline().strip()
                    oneline_dict = dict()
                    if oneline:
                        oneline_json = json.loads(oneline)
                        
                        # 文章 pid
                        pid = oneline_json['id']
                        # 作者 aid, 机构 org_id
                        aid_list = list()
                        authors = oneline_json['authors']
                        for author in authors:
                            aid = author['id']
                            if 'org_id' in author:
                                org_id = author['org_id']
                            else:
                                org_id = ''
                            aid_list.append((aid, org_id))
                        # 出版年
                        if 'year' in oneline_json:
                            year =  oneline_json['year']
                        else:
                            year = ''
                        # 主题 field of study
                        fos_list = list()
                        if 'fos' in oneline_json:
                            fos = oneline_json['fos']
                            for fos_i in fos:
                                fos_list.append(fos_i['name'])                        
                        else:
                            fos = ''
                        # 引文信息
                        if 'references' in oneline_json:
                            ref = oneline_json['references']
                        else:
                            ref = ''
                        # 杂志信息
                        if 'venue' in oneline_json:
                            if 'id' in oneline_json['venue']:
                                venue = oneline_json['venue']['id']
                            else:
                                venue = ''
                        else:
                            venue = ''
                        # 标题信息
                        if 'title' in oneline_json:
                            title = oneline_json['title']
                        else:
                            title = ''
                        
                        oneline_dict["pid"] = pid
                        oneline_dict["aid"] = aid_list
                        oneline_dict["f"]   = fos_list
                        oneline_dict["t"]   = year
                        oneline_dict["r"]   = ref
                        oneline_dict["v"]   = venue
                        oneline_dict['ti']  = title
                        f_output.write(json.dumps(oneline_dict) + "\n")
                    else:
                        break 
    
        f_output.close()
        
        # 清理 zip 文件
        for file_name in os.listdir(dst_dir):
            file_in_path = os.path.join(dst_dir, file_name)
            os.remove(file_in_path)


def ExtractMetaField(file_id, field, file_out_path):
    """ extract papers in the computer science field from the MAG dataset """
    
    file_in_path  = os.path.join(MAG_Meta, "mag_papers_{}.txt".format(file_id))
    file_out_path = file_out_path.format(file_id)
    
    f_output = open(file_out_path, 'w', encoding='utf8')
    with open(file_in_path, 'r') as f:
        while True:
            oneline = f.readline().strip()
            if oneline:
                oneline_json = json.loads(oneline)
                fos = oneline_json['f']
                if field in fos:
                    f_output.write(json.dumps(oneline_json) + "\n")
            else:
                break
    f_output.close()


def ExtractMetaField_MP(field="computer science"):
    """ 多进程调用 ExtractMetaField """
    file_out_dir = os.path.join(FilePath, "MAGv2.1-meta-{}".format(field))

    if not os.path.exists(file_out_dir):
        os.mkdir(file_out_dir)

    file_out_path = os.path.join(file_out_dir, "mag_papers_{}.txt")
    
    print("主进程信息： pid=%s, ppid=%s" % (os.getpid(), os.getppid()))
    ps_list = list()
    for file_id in range(0, 17):
        ps = multiprocessing.Process(target = ExtractMetaField, args = (file_id, field, file_out_path,))
        ps.start()
        print("子进程 ps pid: " + str(ps.pid) + ", ident:" + str(ps.ident))
        ps_list.append(ps)
    
    for ps in ps_list:
        ps.join() # 等待子进程完成任务


def main():
    # OAG2.1 can be downloaded from https://www.aminer.cn/oag-2-1
    # unzip the MAG.zip file downloaded from the above url.
    ExtractMeta()
    # extract papers in the computer science field
    ExtractMetaField_MP()


if __name__ == "__main__":
  main()
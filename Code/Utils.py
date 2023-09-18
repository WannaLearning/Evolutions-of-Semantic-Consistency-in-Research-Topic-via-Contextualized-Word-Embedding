#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:01:55 2023

@author: aixuexi
"""
import pickle

# the propocessed data is saved in this dir
abs_file_path = "/mnt/disk2/MAG_DATA_SET/BERT-cs"

# save/ read file in .pkl file
def save_file(file, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(file, f)
        
def read_file(file_path):
    with open(file_path, "rb") as f:
        file = pickle.load(f)
    return file

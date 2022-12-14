# -*- coding: utf-8 -*-
"""Test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ETEC8JcfAYJHd63vbNA2memWsdtWJ-LN
"""

import numpy as np
import pandas as pd
import os,sys

#files.upload() # needed by google colab

feat_file=sys.argv[1]
go_labels_dir=sys.argv[2]

new_feat_dir="NEW_FEATS"
new_go_dir="NEW_GOS"
try:
    os.makedirs(new_feat_dir)
    os.makedirs(new_go_dir)
except FileExistsError:
    print("WARNING: DIRECTORIES EXIST!")

#go_labels_file=sys.argv[2]
for fname in os.listdir(go_labels_dir):
    #print(fname)
    go_labels_file=go_labels_dir+os.sep+fname

    #arr = np.loadtxt("GO_0003677_DNA_binding.txt", skiprows=1, dtype='str')
    arr = np.loadtxt(go_labels_file, skiprows=1, dtype='str')


    """Separating arrays of ones and zeros, saving the indices from the original array"""

    ind_ones=[]; ind_zeros=[]
    zeros_matrix = np.empty((0,3), str)
    ones_matrix = np.empty((0,3), str)
    for i in range(len(arr)):
      if arr[i,2] == '1':
        ind_ones.append(i)
        ones_matrix = np.vstack((ones_matrix, arr[i]))
      else:
        ind_zeros.append(i)
        zeros_matrix = np.vstack((zeros_matrix, arr[i]))

    """Randomly down-sampling of zeros array"""

    indices = np.random.choice(zeros_matrix.shape[0], len(ind_ones), replace=False)
    zeros_under = zeros_matrix[indices]

    """Number of unique proteins with this function"""

    unique_ones = []
    duplicates_ones = 0
    for i in range (len(ones_matrix)):
      if ones_matrix[i,1] in unique_ones:
        duplicates_ones+=1
      else:
        unique_ones.append(ones_matrix[i,1])
    print("Number of IDRs with this function: ", len(ones_matrix))
    print("Number of unique proteins with this function: ", len(unique_ones))
    print("Number of duplicates: ", duplicates_ones)

    """Number of unique proteins from under-sampled matrix of zeros, that do not have this function"""

    unique_zeros = []
    duplicates_zeros = 0
    for i in range (len(zeros_under)):
      if zeros_under[i,1] in unique_zeros:
        duplicates_zeros+=1
      else:
        unique_zeros.append(zeros_under[i,1])
    print("Size of a test sample of IDRs without this function: ", len(zeros_under))
    print("Number of unique proteins from a test sample without this function: ", len(unique_zeros))
    print("Number of duplicates: ", duplicates_zeros)

    """Creating a new feature and function definition matrices with selected proteins"""

    #features_arr = np.loadtxt("RES_ES_WINDELS_20220525_CLEAN_CAPPED_MEAN_ONLY.out.txt", skiprows=1, dtype='str')
    features_arr = np.loadtxt(feat_file, skiprows=1, dtype='str')
    features_list = np.loadtxt("features_list.txt", dtype='str')
    test_data = np.loadtxt("protein_function_definition.txt", dtype='str')

    unique_sum = unique_zeros + unique_ones
    for i in range(len(unique_sum)):
      for j in range(len(features_arr)):
        if features_arr[j,1] == unique_sum[i]:
          features_list = np.vstack((features_list, features_arr[j,:]))
    df = pd.DataFrame(features_list)
    go_tag=go_labels_file.split(os.sep)[-1][:-4]
    new_feat_file=new_feat_dir+os.sep+go_tag+"_MEAN_Z_FEAT.out.txt"
    df.to_csv(new_feat_file, sep = '\t', encoding='utf-8', header = False, index=False)

    for i in range(len(unique_sum)):
      for j in range(len(arr)):
        if arr[j,1] == unique_sum[i]:
          test_data = np.vstack((test_data, arr[j,:]))
    df = pd.DataFrame(test_data)
    df.to_csv(new_go_dir+os.sep+go_tag+".txt", sep = '\t', encoding='utf-8', header = False, index=False)

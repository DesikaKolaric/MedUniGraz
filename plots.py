import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
from matplotlib.colors import TwoSlopeNorm

##FAIDR PERFORMANCE DATA ANALYSIS
data = pd.read_csv('FAIDR_performance_refined.tsv', sep='\t')
#Number of proteins vs optimal threshold
plt.scatter(data["N_PROTEINS_TRAINING"], data["OPTIMAL THRESHOLD"], s=5, alpha = 0.5)
plt.xlabel("Number of proteins in training")
plt.ylabel("Optimal treshold")
plt.title("FAIDR performance linear analysis")
plt.show()
#Zoom in on functions with less than 750 proteins in training
up_to_750 = data[data["N_PROTEINS_TRAINING"]<750]
plt.scatter(up_to_750["N_PROTEINS_TRAINING"], up_to_750["OPTIMAL THRESHOLD"], s=5, alpha = 0.5)
plt.xlabel("Number of proteins in training")
plt.ylabel("Optimal threshold")
plt.title("FAIDR performance - less than 750 proteins in training")
plt.show()
#Precision vs recall
plt.scatter(data["PRECISION(PPV)"], data["RECALL"], s=5, alpha = 0.5)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("FAIDR performance linear analysis")
plt.show()
#Zoom in on functions with less than 750 proteins in training
plt.scatter(up_to_750["PRECISION(PPV)"], up_to_750["RECALL"], s=5, alpha = 0.5)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("FAIDR performance - less than 750 proteins in training")
plt.show()

###FAIDR analysis with groups of functions colored differently
#functions with recall > 0.7
recall = data[data.RECALL > 0.7]
recall.to_csv('good_recall', sep = '\t', encoding = 'utf-8')
#uploading a new good_recall file with manually added groups of functions
good_recall = pd.read_csv('good_recall_function_groups.txt', sep = '\t')
good_recall_arr = good_recall.to_numpy()
#add a new column of zeros to original faidr performance matrix as a function group label
data['group'] = 0
faidr_matrix = data.to_numpy()
#changing the group number for functions with recall > 0.7 in the main faidr matrix
for i in range(good_recall_arr.shape[0]):
  j = good_recall_arr[i][1]
  faidr_matrix[j][9] = good_recall_arr[i][0]

 #creating a dictionary of colors and assigning a color to every function in main faidr matrix
colors_dict = {0: 'gray', 1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'pink', 6: 'cyan', 7:'magenta', 8: 'purple', 9: 'black'}
colors_list = []
for i in range(faidr_matrix.shape[0]):
  key = faidr_matrix[i][9]
  colors_list.append(colors_dict[key])

##FAIDR performance linear analysis - functions with recall > 0.7
plt.scatter(data["N_PROTEINS_TRAINING"], data["OPTIMAL THRESHOLD"], c = colors_list, s = 5, alpha = 0.5)
plt.xlabel("Number of proteins in training")
plt.ylabel("Optimal threshold")
plt.title("FAIDR performance - designated froups of functions")
plt.show()

plt.scatter(data["PRECISION(PPV)"], data["RECALL"], c = colors_list, s = 5, alpha = 0.5)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("FAIDR performance - designated froups of functions")
plt.show()

##Zooming in on Threshold vs N of proteins where N < 750
up_to= faidr_matrix[faidr_matrix[:,8]<750]
new_colors = []
for i in range(up_to.shape[0]):
  key2 = up_to[i][9]
  new_colors.append(colors_dict[key2])
plt.scatter(up_to[:,8], up_to[:,6], s=2, c = new_colors)
plt.xlabel("Number of proteins in training")
plt.ylabel("Optimal threshold")
plt.title("FAIDR performance - less than 750 proteins in training")
plt.show()

###IDR FEATURES DATA ANALYSIS
#Correlation matrix of features
data2 = pd.read_csv('RES_ES_WINDELS_20220525_CLEAN_CAPPED_MEAN_ONLY.out.txt', sep = '\t')
features = data2.drop(['idr_name','NAME'], axis = 1)
##remove duplicates and self-self pairs
def get_redundant_pairs(features):
    pairs_to_drop = list()
    cols = features.columns
    for i in range (0, features.shape[1]):
        for j in range (0, i+1):
            pairs_to_drop.append((cols[i], cols[j]))
    return pairs_to_drop

def get_correlations(features):
    au_corr = features.corr().unstack()
    labels_to_drop = get_redundant_pairs(features)
    au_corr = au_corr.drop(labels = labels_to_drop).sort_values(ascending = False)
    ###get just the most (un)correlated features
    final_corr = au_corr[(au_corr >= 0.7) | (au_corr <= -0.7) ]
    return final_corr

corrplot = get_correlations(features)
fig,ax = plt.subplots(figsize = (10,5))
norm = TwoSlopeNorm(vmin = -1, vcenter = 0, vmax = 1)
colors = [plt.cm.RdYlGn(norm(c)) for c in corrplot.values]
corrplot.plot.barh(color = colors)
plt.tight_layout()
plt.title("Most (un)correlated features")
plt.show()

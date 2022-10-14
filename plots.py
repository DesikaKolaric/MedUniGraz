import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
from matplotlib.colors import TwoSlopeNorm


data = pd.read_csv('FAIDR_performance_refined.tsv', sep='\t')
#Number of proteins vs optimal treshold
plt.scatter(data["N_PROTEINS_TRAINING"], data["OPTIMAL THRESHOLD"], s=2)
plt.xlabel("Number of proteins in training")
plt.ylabel("Optimal treshold")
plt.show()
#Zoom in on functions with less than 750 proteins in training
up_to_750 = data[data["N_PROTEINS_TRAINING"]<750]
plt.scatter(up_to_750["N_PROTEINS_TRAINING"], up_to_750["OPTIMAL THRESHOLD"], s=2)
plt.xlabel("Number of proteins in training")
plt.ylabel("Optimal treshold")
plt.show()
#Precision vs recall
plt.scatter(data["PRECISION(PPV)"], data["RECALL"], s=2)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.show()
#Zoom in on functions with less than 750 proteins in training
plt.scatter(up_to_750["PRECISION(PPV)"], up_to_750["RECALL"], s=2)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.show()
#Group with recall > 0.7
good_recall = pd.read_csv('good_recall_function_groups.txt', sep='\t')
good_recall_arr=good_recall.to_numpy()


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
fig,ax = plt.subplots(figsize = (15,10))
norm = TwoSlopeNorm(vmin = -1, vcenter = 0, vmax = 1)
colors = [plt.cm.RdYlGn(norm(c)) for c in corrplot.values]
corrplot.plot.barh(color = colors)
plt.tight_layout()
plt.show()

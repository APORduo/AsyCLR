import torch
import torch.nn as nn
import numpy as  np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity,cosine_distances
from pdb import set_trace
from scipy.spatial.distance import cdist
def intra_dist(feats,labels):
    feats_norm = normalize(feats,axis=1)
    intra_class_dists = []
    for i in np.unique(labels):
        class_points = feats_norm[labels == i]
        intra_class_dist = cosine_distances(class_points,class_points).mean()
        intra_class_dists.append(intra_class_dist)

    
    return np.mean(intra_class_dists)

def inter_dist(feats,labels):
    feats_norm = normalize(feats)
    mean_list = []
    for i in np.unique(labels):
        class_points = feats_norm[labels == i]
        class_mean = class_points.mean(axis=0,keepdims=True)
        mean_list.append(class_mean)
    
    mean_arr = np.concatenate(mean_list,axis=0)
    
    dist_mat =cosine_distances(mean_arr,mean_arr)
    
    class_num = len(mean_list)
    inter = dist_mat.sum()/class_num/(class_num-1)
    return inter
    
def inter_cdist(feats,labels):

    feats_norm = normalize(feats)
    labels = labels.reshape(-1,1)
    mask = labels == labels.T
    mask = 1- mask 
    dist_mat = cosine_distances(feats_norm,feats_norm)
    
    dist_mask = mask * dist_mat
    inter = dist_mask.sum()/mask.sum()
    return inter     
    
    
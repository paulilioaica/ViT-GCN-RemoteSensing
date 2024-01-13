import scipy.io 
from random import sample
import torch
import numpy as np
import cv2
import os
import random 
from random import sample
from torch_geometric.data import Data
import scipy.sparse.csgraph as csgraph

def normalize(A):
    A = np.double(A)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return normalized

def load_data():
    mat = scipy.io.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
    mat = np.array(mat, dtype= np.double)
    for i in range(mat.shape[2]):
        mat[:, :, i] = normalize(mat[:, :, i])
    mat_gt = scipy.io.loadmat('Indian_pines_gt.mat')['indian_pines_gt']



    train_indx = []
    test_indx = []
    for value in range(1, 17):
        x, y = np.where(mat_gt == value)
        label_coord = [(i, j) for i, j in zip(x,y)]
        random.shuffle(label_coord)
        train_dim = int(0.3 * len(label_coord))
        train_indx.extend(label_coord[:train_dim])
        test_indx.extend(label_coord[train_dim:])
    return mat, mat_gt, train_indx, test_indx

def generate_training_graphs(mat, indexes, sample_size=189):
    for z in range(250):
        sampled_indexes = sample(indexes, k = sample_size)
        slices = []
        A_train = np.zeros((len(sampled_indexes), len(sampled_indexes)))
        for i in range(len(sampled_indexes)):
            for j in range(len(sampled_indexes)):
                A_train[i][j] = np.exp(- (np.linalg.norm(mat[sampled_indexes[i]] - mat[sampled_indexes[j]]) ** 2) / 1)
        
        #graph
        L_train = csgraph.laplacian(A_train)
        x, y = np.where(L_train != 0.0)
        edge_index = torch.tensor([[i, j] for i, j in zip(x,y)], dtype=torch.int32)
        edge_features = torch.tensor([L_train[i][j] for i, j in zip(x,y)], dtype=torch.float16)
        x = torch.tensor([mat[i][j] for i,j in sampled_indexes], dtype=torch.float16)
        y = torch.tensor([mat_gt[i][j] for i,j in sampled_indexes], dtype=torch.int8) - 1

        #conv
        for i, j in sampled_indexes:
          slice_ = np.zeros((5, 5, 200))
          i_start = i - 2
          i_end = i + 2
          j_start = j - 2
          j_end = j + 2
          slice_temp = mat[i_start:i_end, j_start:j_end, :]
          slice_[:slice_temp.shape[0], :slice_temp.shape[1], :] = slice_temp
          slices.append(slice_)

        slices = torch.tensor(np.array(slices), dtype=torch.float16)
        graph = Data(x=x, edge_index=edge_index.T, edge_features=edge_features, y=y)
        graph.slices = slices
        torch.save(graph, os.path.join("./", "train_graph_" + str(z) + ".pt"))
        print("Saved graph " + "train_graph_" + str(z) + ".pt")


def generate_training_graphs_1(mat, indexes, sample_size=189):
    z = 0
    while len(indexes) >= 189:
        print(len(indexes))
        slices = []
        sampled_indexes = sample(indexes, k = sample_size)
        A_train = np.zeros((len(sampled_indexes), len(sampled_indexes)))
        for sample_ in sampled_indexes:
            indexes.remove(sample_)
        for i in range(len(sampled_indexes)):
            for j in range(len(sampled_indexes)):
                A_train[i][j] = np.exp(- (np.linalg.norm(mat[sampled_indexes[i]] - mat[sampled_indexes[j]]) ** 2) / 1)
        
        #graph
        L_train = csgraph.laplacian(A_train)
        x, y = np.where(L_train != 0)
        edge_index = torch.tensor([[i, j] for i, j in zip(x,y)], dtype=torch.int32)
        edge_features = torch.tensor([L_train[i][j] for i, j in zip(x,y)], dtype=torch.float16)
        x = torch.tensor([mat[i][j] for i,j in sampled_indexes], dtype=torch.float16)
        y = torch.tensor([mat_gt[i][j] for i,j in sampled_indexes], dtype=torch.int8) - 1
        graph = Data(x=x, edge_index=edge_index.T, edge_features=edge_features, y=y)

    
        #conv
        for i, j in sampled_indexes:
          slice_ = np.zeros((5, 5, 200))
          i_start = i - 2
          i_end = i + 2
          j_start = j - 2
          j_end = j + 2
          slice_temp = mat[i_start:i_end, j_start:j_end, :]
          slice_[:slice_temp.shape[0], :slice_temp.shape[1], :] = slice_temp
          slices.append(slice_)

        slices = torch.tensor(np.array(slices), dtype=torch.float16)
        graph = Data(x=x, edge_index=edge_index.T, edge_features=edge_features, y=y)
        graph.slices = slices

        torch.save(graph, os.path.join("./", "train_graph_" + str(z) + ".pt"))
        print("Saved graph " + "train_graph_" + str(z) + ".pt")
        z += 1





def generate_test_graphs(mat, indexes, sample_size=189):
    z = 0
    while len(indexes) > 0:
        print(len(indexes))
        slices = []
        sampled_indexes = sample(indexes, k = sample_size)
        A_train = np.zeros((len(sampled_indexes), len(sampled_indexes)))
        for sample_ in sampled_indexes:
            indexes.remove(sample_)
        for i in range(len(sampled_indexes)):
            for j in range(len(sampled_indexes)):
                A_train[i][j] = np.exp(- (np.linalg.norm(mat[sampled_indexes[i]] - mat[sampled_indexes[j]]) ** 2) / 1)
        
        #graph
        L_train = csgraph.laplacian(A_train)
        x, y = np.where(L_train != 0.000002313)
        edge_index = torch.tensor([[i, j] for i, j in zip(x,y)], dtype=torch.int32)
        edge_features = torch.tensor([L_train[i][j] for i, j in zip(x,y)], dtype=torch.float16)
        x = torch.tensor([mat[i][j] for i,j in sampled_indexes], dtype=torch.float16)
        y = torch.tensor([mat_gt[i][j] for i,j in sampled_indexes], dtype=torch.int8) - 1
        graph = Data(x=x, edge_index=edge_index.T, edge_features=edge_features, y=y)

    
        #conv
        for i, j in sampled_indexes:
          slice_ = np.zeros((5, 5, 200))
          i_start = i - 2
          i_end = i + 2
          j_start = j - 2
          j_end = j + 2
          slice_temp = mat[i_start:i_end, j_start:j_end, :]
          slice_[:slice_temp.shape[0], :slice_temp.shape[1], :] = slice_temp
          slices.append(slice_)

        slices = torch.tensor(np.array(slices), dtype=torch.float16)
        graph = Data(x=x, edge_index=edge_index.T, edge_features=edge_features, y=y)
        graph.slices = slices
        print(sampled_indexes)
        graph.idx = torch.tensor(sampled_indexes)
        
        torch.save(graph, os.path.join("./", "test_graph_" + str(z) + ".pt"))
        print("Saved graph " + "test_graph_" + str(z) + ".pt")
        z += 1

generate_training_graphs(mat, train_indx)

generate_test_graphs(mat, test_indx)
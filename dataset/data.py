import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import numpy as np
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import scipy.sparse as sp

def load_data():
    PATH = r'E:\code\grape\PHMGNNBenchmark-main\process\GCN_imgs'

    print('Loading data...')
    class_list = ["ball_18_imgs", "ball_36_imgs", "ball_54_imgs",
                  "inner_18_imgs", "inner_36_imgs", "inner_54_imgs",
                  "outer_18_imgs", "outer_36_imgs", "outer_54_imgs", "normal_imgs"]

    feature_matrices = []  # np.zeros((len(df), N, 1))
    adj_matrices = []  # np.zeros((len(df), N, N))
    labels = []  # np.zeros((len(df), 1))
    nums = []
    dataset = []  # data数据对象的list集合
    for i in tqdm(range(len(class_list))):
        cls_dir = os.path.join(PATH, class_list[i])
        list = os.listdir(cls_dir)
        for i in tqdm(range(len(list))):
            labels.append(i)

            edge_index_temp = sp.coo_matrix(adj_matrices[i])
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            edge_index = torch.LongTensor(indices)

            # 节点及节点特征数据转换[20,1]
            x = np.array(list(feature_matrices[i].values()))
            x = x.squeeze(0)
            x = torch.FloatTensor(x)

            # 图标签数据转换
            y = torch.LongTensor(labels[i])

            # 构建数据集:为一张图，20个节点，每个节点一个特征，Coo稀疏矩阵的边，一个图一个标签
            data = Data(x=x, edge_index=edge_index, y=y)  # 构建新型data数据对象
            dataset.append(data)  # # 将每个data数据对象加入列表


    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    one_hot_feature_matrices = enc.fit_transform(feature_matrices)
    # one_hot_feature_matrices = np.reshape(one_hot_feature_matrices, (-1, N, 8))
    dataset = []
    for i in range(len(labels)):
        X = torch.from_numpy(one_hot_feature_matrices[i]).float()
        #这里的data.A的由来
        A = torch.from_numpy(adj_matrices[i]).float()
        y = torch.Tensor([[labels[i]]]).float()
        mol_num = torch.Tensor([nums[i]])
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        # breakpoint()
        dataset.append(Data(x=X,
                            edge_index=edge_index,
                            edge_attr=edge_weight,
                            y=y,
                            # smiles=smiles_list[i],
                            A=A,
                            # atomic_nums=feature_matrices[i],
                            mol_num=mol_num
                            ))

    return dataset

if __name__ == '__main__':
    load_data()
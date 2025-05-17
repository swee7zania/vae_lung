import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class LoadData(Dataset): 
    def __init__(self, x, y):
        super(LoadData, self).__init__()
        # store the raw tensors
        self._x = x
        self._y = y

    def __len__(self):
        # a dataset must know its size
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index, :]
        y = self._y[index, :]
        return x, y


def load_latent_vectors(run, results_path):
    """
    加载潜在向量文件 latent_vectors_{Run}.npy
    """
    latent_file = os.path.join(results_path, f"latent_vectors_{run}.npy")
    if not os.path.exists(latent_file):
        raise FileNotFoundError(f"{latent_file} not found. Ensure VAE model is trained.")
    return np.load(latent_file, allow_pickle=True)


def data_split(n, meta, latent_vectors, labels, batch_size):
    """
    数据集分割 - 交叉验证划分
    """
    def which_set(row, data_split):
        for i, dataset in enumerate(data_split):
            if row in dataset:
                return i

    random.seed(42)
    patient_id = list(np.unique(meta['patient_id']))
    data_split, used = [], []

    for i in range(n):
        temp_set = []
        while len(temp_set) < len(patient_id) // n:
            index = random.choice(patient_id)
            if index not in used:
                used.append(index)
                temp_set.append(index)

        if i == n - 1:
            for pat_id in patient_id:
                if pat_id not in used:
                    temp_set.append(pat_id)

        data_split.append(temp_set)

    meta['data_split'] = meta['patient_id'].apply(lambda row: which_set(row, data_split))

    cross_val_data, cross_val_labels = [], []

    for i in range(n):
        vecs, labs = [], []
        for index, item in enumerate(meta['data_split']):
            if item == i:
                vecs.append(torch.tensor(latent_vectors[index]))
                labs.append(torch.tensor(labels[index]))

        vecs = torch.stack(vecs)
        labs = torch.unsqueeze(torch.stack(labs), 1)
        cross_val_data.append(vecs)
        cross_val_labels.append(labs)

    return cross_val_data, cross_val_labels


def get_data_loaders(run, n, meta_file, results_path, batch_size):
    """
    获取数据加载器 (train_loader, val_loader)
    """
    latent_vectors = load_latent_vectors(run, results_path)

    # 加载标签
    meta = pd.read_csv(meta_file)
    labels = meta["label"].values

    # 交叉验证分割
    cross_val_data, cross_val_labels = data_split(n, meta, latent_vectors, labels, batch_size)

    data_loaders = []

    for i in range(n):
        # 获取训练数据 (除第 i 折外的数据)
        train_data, train_labels = [], []
        for j in range(n):
            if j != i:
                train_data.append(cross_val_data[j])
                train_labels.append(cross_val_labels[j])

        train_data = torch.cat(train_data, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        # 构建 DataLoader
        train_dataset = LoadData(train_data, train_labels)
        val_dataset = LoadData(cross_val_data[i], cross_val_labels[i])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        data_loaders.append((train_loader, val_loader))

    return data_loaders

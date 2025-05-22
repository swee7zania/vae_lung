import random
import numpy as np
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

def data_split(n, meta, latent_vectors, labels, batch_size):
    def which_set(row,data_split):
        for i, dataset in enumerate(data_split):
            if row in dataset:
                return i
    random.seed(42)
    patient_id = list(np.unique(meta['patient_id']))
    data_split, used = [], []
    for i in range(n):
        temp_set = []
        while len(temp_set) < len(patient_id)//n:
            index = random.choice(patient_id)
            if index not in used:
                used.append(index)
                temp_set.append(index)
        if i == n-1:
            for pat_id in patient_id:
                if pat_id not in used:
                    temp_set.append(pat_id)    
        data_split.append(temp_set)
        
    
    meta['data_split'] = meta['patient_id'].apply(lambda row : which_set(row,data_split))
    print(len(latent_vectors), len(labels))
    split = list(meta["data_split"])
    cross_val_data, cross_val_labels = [], []
    for i in range(n):
        vecs, labs = [], []
        for index, item in enumerate(split):
            if item == i:
                vecs.append(torch.tensor(latent_vectors[index]))
                labs.append(torch.tensor(labels[index]))            
        vecs = torch.stack(vecs)
        labs = torch.unsqueeze(torch.stack(labs), 1)       
        cross_val_data.append(vecs)
        cross_val_labels.append(labs)
    
    return cross_val_data, cross_val_labels


def Cross_Validation(run, n, meta, latent_vectors, labels, batch_size):
    def other_index(exclude, n):
        index = []
        for i in range(n):
            if i not in exclude:
                index.append(i)
        return index
    def find_subsets(run, n):
        if run != n-1:
            return other_index([n-2-run, n-1-run], n), n-2-run, n-1-run
        if run == n-1:
            return other_index([0, run], n), run, 0

    def concat_train_data(indices, datasets):
        train_data = []
        for idx in indices:
            train_data.append(datasets[idx])
        return train_data
    
    #loss_list, accuracy_list, results_list, auc_list = [], [], [], []
    cross_val_data, cross_val_labels = data_split(n, meta, latent_vectors, labels, batch_size)    
    
    train_data, train_labels = [], []
    cross_val_split = find_subsets(run, n)
    for i in cross_val_split[0]:
        train_data.append(cross_val_data[i])
        train_labels.append(cross_val_labels[i])
    train_data = torch.cat(train_data,dim=0)
    train_labels = torch.cat(train_labels,dim=0)

    train_dataset = LoadData(train_data, train_labels)
    val_index = cross_val_split[1]
    test_index = cross_val_split[2]
    validation_dataset = LoadData(cross_val_data[val_index], cross_val_labels[val_index])
    test_dataset = LoadData(cross_val_data[test_index], cross_val_labels[test_index])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False) 
    
    return train_loader, validation_loader, test_loader
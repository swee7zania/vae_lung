import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Custom PyTorch Dataset class that encapsulates features (x) and labels (y)
# 自定义 PyTorch 数据集类，用于封装特征（x）和标签（y）
class LoadData(Dataset): 
    def __init__(self, x, y):
        super(LoadData, self).__init__()
        self._x = x     # Store latent vectors 存储潜变量特征
        self._y = y     # Store labels 存储标签

    # Return the number of samples 返回样本数量
    def __len__(self):
        return self._x.shape[0]

    # Return one sample 返回一个样本（特征 + 标签）
    def __getitem__(self, index):
        x = self._x[index, :]
        y = self._y[index, :]
        return x, y


# Used to divide patient data into n folds by patient ID and return the data and labels for each fold
# 用于将病人数据按病人 ID 分为 n 折，返回每折的数据和标签
def data_split(n, meta, latent_vectors, labels, batch_size):
    # Which fold a patient belongs to 某个病人 ID 属于哪个折
    def which_set(row,data_split):
        for i, dataset in enumerate(data_split):
            if row in dataset:
                return i
    
    # Fix the random seed to ensure that the partitioning results are reproducible
    # 固定随机种子，确保划分结果可复现
    random.seed(42)
    
    # Get all unique patient IDs 获取所有唯一的病人 ID
    patient_id = list(np.unique(meta['patient_id']))
    
    data_split, used = [], []
    
    # Randomly and evenly distribute patient IDs into n sets
    # 将病人 ID 随机均匀分配到 n 个集合中
    for i in range(n):
        temp_set = []
        while len(temp_set) < len(patient_id)//n:
            index = random.choice(patient_id)
            if index not in used:
                used.append(index)
                temp_set.append(index)
        # The last fold to fill the remaining patients 最后一折补齐剩余病人
        if i == n-1:
            for pat_id in patient_id:
                if pat_id not in used:
                    temp_set.append(pat_id)    
        data_split.append(temp_set)
    
    # Add a new column in meta to mark the fold index to which each sample belongs
    # 在 meta 中新增列，标记每个样本所属的fold索引
    meta['data_split'] = meta['patient_id'].apply(lambda row : which_set(row,data_split))
    print(len(latent_vectors), len(labels))
    
    # Get the fold index list to which each sample belongs 获取每个样本所属的fold索引列表
    split = list(meta["data_split"])
    
    # Store the divided n-fold latent variables and labels 存储划分好的 n 折潜变量和标签
    cross_val_data, cross_val_labels = [], []
    
    for i in range(n):
        vecs, labs = [], []
        for index, item in enumerate(split):
            if item == i:
                vecs.append(torch.tensor(latent_vectors[index]))   # 转为 tensor 格式
                labs.append(torch.tensor(labels[index]))            
        vecs = torch.stack(vecs)                         # 合并成一个tensor
        labs = torch.unsqueeze(torch.stack(labs), 1)     # 标签升维为 (N, 1)
        cross_val_data.append(vecs)
        cross_val_labels.append(labs)
    
    return cross_val_data, cross_val_labels

# Generate training, validation, and test loaders for a given fold
# 根据当前折数 run，返回该折的训练集、验证集和测试集的 DataLoader
def Cross_Validation(run, n, meta, latent_vectors, labels, batch_size):
    # Get all indices except the excluded ones 获取除指定索引之外的其他索引
    def other_index(exclude, n):
        index = []
        for i in range(n):
            if i not in exclude:
                index.append(i)
        return index
    
    # Define train/val/test splits based on current run 根据当前运行定义训练/验证/测试分割
    def find_subsets(run, n):
        # Example (5-fold, run=1): training set=[0,2,3] validation=2 test=3
        # 示例（5折，run=1）：训练集=[0,2,3] 验证=2 测试=3
        if run != n-1:
            return other_index([n-2-run, n-1-run], n), n-2-run, n-1-run
        
        # When run is the last fold, manually specify the validation set and test set
        # run是最后一折时，手动指定验证集和测试集
        if run == n-1:
            return other_index([0, run], n), run, 0
    
    # Concatenate multiple folds of training data into a complete training set
    # 将多折训练数据拼接为完整训练集
    def concat_train_data(indices, datasets):
        train_data = []
        for idx in indices:
            train_data.append(datasets[idx])
        return train_data
    
    # Step 1: Perform intra-fold data partitioning (patient grouping → data/labels)
    # Step 1: 执行折内数据划分（病人分组 → 数据/标签）
    cross_val_data, cross_val_labels = data_split(n, meta, latent_vectors, labels, batch_size)    
    
    # Step 2: Get the training/validation/test fold index of the current run
    # Step 2: 获取当前run的训练/验证/测试折索引
    train_data, train_labels = [], []
    cross_val_split = find_subsets(run, n)
    
    # Concatenate multiple fold training sets 把多折训练集拼接起来
    for i in cross_val_split[0]:
        train_data.append(cross_val_data[i])
        train_labels.append(cross_val_labels[i])
    
    # Concatenate all training features 拼接所有训练特征
    train_data = torch.cat(train_data,dim=0)
    # Concatenate all training labels 拼接所有训练标签
    train_labels = torch.cat(train_labels,dim=0)

    # Step 3: Build a Dataset object and pass it to DataLoader
    # Step 3: 构建Dataset对象并传入DataLoader
    train_dataset = LoadData(train_data, train_labels)
    
    # The validation set index of the current fold
    # 当前折的验证集索引
    val_index = cross_val_split[1]
    # The test set index of the current fold
    # 当前折的测试集索引
    test_index = cross_val_split[2]
    
    validation_dataset = LoadData(cross_val_data[val_index], cross_val_labels[val_index])
    test_dataset = LoadData(cross_val_data[test_index], cross_val_labels[test_index])
    
    # Step 4: Build DataLoader for model training
    # Step 4: 构建DataLoader，供模型训练使用
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False) 
    
    return train_loader, validation_loader, test_loader
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define custom dataset class 定义自定义数据集类
class LoadImages(Dataset):
    def __init__(self, main_dir, files_list, HU_Upper, HU_Lower):
        self.main_dir = main_dir      # Image directory 设置图像目录
        self.all_imgs = files_list    # List of image file names 图像文件名列表
        self.transform = transforms.Compose([transforms.ToTensor()])  # Image to tensor 转换图像为张量
        self.HU_Upper = HU_Upper      # Hounsfield Unit bound HU上限
        self.HU_Lower = HU_Lower      # Hounsfield Unit bound HU下限
    
    # Return total number of images 返回图像总数
    def __len__(self):
        return len(self.all_imgs)
    
    # Read and transformed image 读取及转换图像
    def __getitem__(self, index):
        img_loc = os.path.join(self.main_dir, self.all_imgs[index])
        img = np.load(img_loc)
        
        # Normalize image values based on HU window 基于HU范围对图像值进行归一化
        img = np.where((self.HU_Lower <= img) & (img <= self.HU_Upper), (img - self.HU_Lower) / (self.HU_Upper - self.HU_Lower), img)
        img[img < self.HU_Lower] = 0    # Set lower values to 0 将低于下限的值设为0
        img[img > self.HU_Upper] = 1    # Set higher values to 1 将高于上限的值设为1
        img = self.transform(img)       # Convert to tensor 转换为张量
        return img

# Split data into training and testing sets 将数据划分为训练集和测试集
def vae_data_split(IMAGE_DIR, meta_file, all_files_list, batch_size, HU_UpperBound, HU_LowerBound):
    meta = pd.read_csv(meta_file)
    def is_train(row,train,test):
        if row in train:
            return 'Train'
        else:
            return 'Test'
    
    # Get split labels 获取划分标签
    split = list(meta["data_split"])
    train_images, test_images = [], []
    
    # Assign images to train/test lists 根据标签划分图像到训练或测试列表
    for index, item in enumerate(split):
        if item == 'Train':
            train_images.append(all_files_list[index])
        if item == 'Test':
            test_images.append(all_files_list[index])
            
    print("Samples:     Train:", len(train_images), "   Test:", len(test_images))
    
    # Load image datasets for train and test 加载训练和测试图像数据集
    train_images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=train_images, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound)
    test_images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=test_images, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound)
    
    # Create DataLoaders 创建数据加载器
    train_loader = DataLoader(train_images, batch_size, shuffle=True)
    test_loader = DataLoader(test_images, batch_size, shuffle=False)
    
    return train_loader, test_loader
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LoadImages(Dataset):
    def __init__(self, main_dir, files_list, HU_Upper, HU_Lower):
        self.main_dir = main_dir
        self.all_imgs = files_list
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.HU_Upper = HU_Upper
        self.HU_Lower = HU_Lower

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        img_loc = os.path.join(self.main_dir, self.all_imgs[index])
        img = np.load(img_loc)
        img = np.where((self.HU_Lower <= img) & (img <= self.HU_Upper), (img - self.HU_Lower) / (self.HU_Upper - self.HU_Lower), img)
        img[img < self.HU_Lower] = 0
        img[img > self.HU_Upper] = 1
        img = self.transform(img)
        return img

def vae_data_split(IMAGE_DIR, meta_file, all_files_list, batch_size, HU_UpperBound, HU_LowerBound):
    meta = pd.read_csv(meta_file)
    def is_train(row,train,test):
        if row in train:
            return 'Train'
        else:
            return 'Test'
    patient_id = list(np.unique(meta['patient_id']))
    train_patient , test_patient = train_test_split(patient_id,test_size = 0.3)
    meta['data_split']= meta['patient_id'].apply(lambda row : is_train(row,train_patient,test_patient))

    split = list(meta["data_split"])
    train_images, test_images = [], []
    for index, item in enumerate(split):
        if item == 'Train':
            train_images.append(all_files_list[index])
        if item == 'Test':
            test_images.append(all_files_list[index])
            
    print("Samples:     Train:", len(train_images), "   Test:", len(test_images))
    print("Proportions:       {: .3f},      {: .3f}".format(100*len(train_images)/13852, 100*len(test_images)/13852))
    train_images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=train_images, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound)
    test_images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=test_images, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound)
    train_loader = DataLoader(train_images, batch_size, shuffle=True)
    test_loader = DataLoader(test_images, batch_size, shuffle=False)
    return train_loader, test_loader
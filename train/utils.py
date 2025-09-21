import os
import numpy as np
import pandas as pd
import ast
from torch.utils.data.dataset import Dataset
import torch
from monai.transforms import *

TARGET_SIZE = (224, 320, 224)

basic_transforms = Compose([
    RandZoom(prob=0.75, min_zoom=0.75, max_zoom=1.25),
    RandGaussianNoise(prob=0.75, mean=0, std=0.01),
    # RandHistogramShift(num_control_points=10, prob=0.2),
    RandAxisFlip(prob=0.5),
    RandRotate90(prob=0.3, spatial_axes=(0, 2)),
    RandRotate90(prob=0.3, spatial_axes=(0, 1)),
    RandRotate90(prob=0.3, spatial_axes=(1, 2)),
    Resize((224, 320, 224)),
])


class TrainSetLoader_classification(Dataset):
    def __init__(self, info_path, folder=0, train=True):
        super(TrainSetLoader_classification, self).__init__()
        self.info = np.array(pd.read_excel(info_path,
                                           sheet_name="B&G_2"))
        self.train = train

        self.train_list = [item for item in self.info[:, 0] if "Covid" in item]

        if self.train:
            self.train_list = self.info[self.info[:, -1] != folder][:, 0]
        else:
            self.train_list = self.info[self.info[:, -1] == folder][:, 0]

    def __getitem__(self, index):
        loc = np.where(self.info[:, 0] == self.train_list[index])[0][0]

        try:
            array = np.load(self.train_list[index], allow_pickle=True)
            data = array["data"][np.newaxis]
        except:
            array = np.load(self.train_list[index], allow_pickle=True)
            data = array["arr_0"][np.newaxis]

        data = np.clip((data + 1000.0) / 1600, 0, 1)

        if self.train:
            data = basic_transforms(data)

        data = basic_transforms(data)

        data = torch.tensor(data).to(torch.float)
        info = np.array(self.info[loc, 2:12], "float32")
        info[-1] /= 100
        info[:4] = info[:4] / self.info[loc, 1] * 10

        label = int(self.info[loc, -2])
        if label in [0, 1]:
            true_label = 0
            hot_label = [1, 0, 0]
        elif label == 2:
            true_label = 1
            hot_label = [0, 1, 0]
        else:
            true_label = 2
            hot_label = [0, 0, 1]

        info = torch.tensor(info, dtype=torch.float)

        if not self.train:
            label = torch.tensor(int(true_label), dtype=torch.float)
            return data, info, label

        label = torch.tensor(hot_label, dtype=torch.float)
        return data, info, label

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader_OI(Dataset):
    def __init__(self, info_path, lesion_path, folder=0, train=True):
        super(TrainSetLoader_OI, self).__init__()
        # self.dataset_dir = dataset_dir
        self.info = np.array(pd.read_excel(info_path,
                                           sheet_name="B&G_2"))

        self.pneumonia = np.array(pd.read_excel(lesion_path,
                                                sheet_name="Pneumonia_2"))

        self.train = train

        if self.train:
            self.train_list = self.info[self.info[:, -1] != folder][:, 0]
        else:
            self.train_list = self.info[self.info[:, -1] == folder][:, 0]
            self.train_list = [item for item in self.train_list if "ARDS_2" not in item]

    def __getitem__(self, index):
        loc = np.where(self.info[:, 0] == self.train_list[index])[0][0]
        loc_pneumonia = np.where(self.pneumonia[:, 0] == self.train_list[index])[0][0]

        array = np.load(self.train_list[index], allow_pickle=True)
        data = array["data"][np.newaxis]

        data = np.clip((data + 1000.0) / 1600, 0, 1)

        if self.train:
            data = basic_transforms(data)

        pao2 = self.info[loc, 8] / 100
        fio2 = self.info[loc, 9] / 100
        # oi = pao2 / fio2

        oi = self.info[loc, 11] / 100

        age = self.info[loc, 6] / 100

        pneumonia_info = self.pneumonia[loc_pneumonia, 1:10]
        pneumonia_info[1:5] = pneumonia_info[1:5] / pneumonia_info[0] * 10
        pneumonia_info = pneumonia_info[1:]

        info_list = [fio2, age]
        info_list.extend(pneumonia_info)
        # print(self.train_list[index], info_list)

        return (torch.tensor(data).to(torch.float),
                torch.tensor([oi]), torch.tensor(info_list))

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader_prognosis(Dataset):
    def __init__(self, info_path, lesion_path, folder=0, train=True):
        super(TrainSetLoader_prognosis, self).__init__()

        self.info = np.array(pd.read_excel(info_path,
                                           sheet_name="Prognosis"))
        self.pneumonia = np.array(pd.read_excel(lesion_path,
                                                sheet_name="Pneumonia_2"))
        self.train = train
        self.info[:, 3] = np.abs(self.info[:, 3])

        if self.train:
            self.train_list = self.info[self.info[:, -1] != folder][:, 0]
        else:
            self.train_list = self.info[self.info[:, -1] == folder][:, 0]

    def __getitem__(self, index):
        loc = np.where(self.info[:, 0] == self.train_list[index])[0][0]
        loc_pneumonia = np.where(self.pneumonia[:, 0] == self.train_list[index])[0][0]

        try:
            array = np.load(self.train_list[index], allow_pickle=True)
            data = array["data"][np.newaxis]
        except:
            array = np.load(self.train_list[index], allow_pickle=True)
            data = array["arr_0"][np.newaxis]

        data = np.clip((data + 1000.0) / 1600, 0, 1)

        if self.train:
            data = basic_transforms(data)

        data = torch.tensor(data).to(torch.float)

        label = self.info[loc, 3]

        age = self.pneumonia[loc_pneumonia, -2] / 100
        pneumonia_info = self.pneumonia[loc_pneumonia, 1:10]
        pneumonia_info[1:5] = pneumonia_info[1:5] / pneumonia_info[0]
        pneumonia_info = pneumonia_info[1:]

        info_list = [age]
        info_list.extend(pneumonia_info)

        if not self.train:
            label = torch.tensor(int(label), dtype=torch.float).cuda()
            return data, label, torch.tensor(info_list)

        label = torch.tensor([1 - label, label], dtype=torch.float).cuda()
        return data, label, torch.tensor(info_list)

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader_OI_2(Dataset):
    def __init__(self, info_path, lesion_path, folder=0, train=True):
        super(TrainSetLoader_OI_2, self).__init__()
        # self.dataset_dir = dataset_dir
        self.info = np.array(pd.read_excel(info_path,
                                           sheet_name="B&G_2"))

        self.pneumonia = np.array(pd.read_excel(lesion_path,
                                                sheet_name="Pneumonia_2"))

        self.train = train

        if self.train:
            self.train_list = self.info[self.info[:, -1] != folder][:, 0]
        else:
            self.train_list = self.info[self.info[:, -1] == folder][:, 0]

    def __getitem__(self, index):
        loc = np.where(self.info[:, 0] == self.train_list[index])[0][0]
        loc_pneumonia = np.where(self.pneumonia[:, 0] == self.train_list[index])[0][0]

        array = np.load(self.train_list[index], allow_pickle=True)
        data = array["data"][np.newaxis]

        data = np.clip((data + 1000.0) / 1600, 0, 1)

        if self.train:
            data = basic_transforms(data)

        pao2 = self.info[loc, 8] / 100
        fio2 = self.info[loc, 9] / 100
        oi = pao2 / fio2

        if self.train:
            if np.random.random() < 0.8:
                fio2_k = fio2
                oi_range = [oi, oi]
            else:
                fio2_k = np.random.uniform(low=0.21, high=1)
                oi_estimate = pao2 / fio2_k
                oi_range = [min(oi_estimate, oi), max(oi_estimate, oi)]
        else:
            fio2_k = fio2
            oi_range = [oi, oi]

        age = self.info[loc, 6] / 100

        pneumonia_info = self.pneumonia[loc_pneumonia, 1:10]
        pneumonia_info[1:5] = pneumonia_info[1:5] / pneumonia_info[0] * 10
        pneumonia_info = pneumonia_info[1:]

        info_list = [fio2_k, age]
        info_list.extend(pneumonia_info)
        # print(self.train_list[index], info_list)

        return (torch.tensor(data).to(torch.float),
                torch.tensor(oi_range), torch.tensor(info_list))

    def __len__(self):
        return len(self.train_list)


class TrainSetLoader_survival(Dataset):
    def __init__(self, info_path, lesion_path, folder=0, train=True):
        super(TrainSetLoader_survival, self).__init__()
        # self.dataset_dir = dataset_dir
        self.info = np.array(pd.read_excel(info_path,
                                           sheet_name="Prognosis_2"))
        self.pneumonia = np.array(pd.read_excel(lesion_path,
                                                sheet_name="Pneumonia_2"))
        self.train = train

        if self.train:
            self.train_list = self.info[(self.info[:, 1] % 6) != folder][:, 0]
        else:
            self.train_list = self.info[(self.info[:, 1] % 6) == folder][:, 0]

    def generate_death_status_and_mask(self, during, event, max_days=28):
        death_status = []
        mask = []

        for t in range(0, max_days + 1):
            if event == 1 and t >= during:
                death_status.append(1)
            else:
                death_status.append(0)

            if event == 0 and t > during:
                mask.append(0)
            else:
                mask.append(1)

        return death_status, mask

    def __getitem__(self, index):
        loc = np.where(self.info[:, 0] == self.train_list[index])[0][0]

        try:
            array = np.load(self.train_list[index], allow_pickle=True)
            data = array["data"][np.newaxis]
        except:
            array = np.load(self.train_list[index], allow_pickle=True)
            data = array["arr_0"][np.newaxis]

        data = np.clip((data + 1000.0) / 1600, 0, 1)

        if self.train:
            data = basic_transforms(data)

        data = torch.tensor(data).to(torch.float)

        during = self.info[loc, 3] + 1
        event = self.info[loc, 4]

        death_status, mask = self.generate_death_status_and_mask(during, event)

        age = self.info[loc, 1] / 100
        sex = self.info[loc, 2] / 100

        info_list = [age, sex]
        # info_list.extend(pneumonia_info)

        death_status = torch.tensor(death_status, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        return data, death_status, mask, torch.tensor(info_list)

    def __len__(self):
        return len(self.train_list)

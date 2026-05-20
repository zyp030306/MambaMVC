from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py


class mnist4():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/mnist4.mat')
        self.Y = np.array(data['Y']).astype(np.int32).reshape(4000, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        # self.V4 = data['X'][0][3].astype(np.float32)
        # self.V5 = data['X'][0][4].astype(np.float32)
        # self.V6 = data['X'][0][5].astype(np.float32)
        # self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 4000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        # x4 = self.V4[idx]
        # x5 = self.V5[idx]
        # x6 = self.V6[idx]
        # x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        # x4 = self.V4
        # x5 = self.V5
        # x6 = self.V6
        # # x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y


class synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 600

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y



class BDGP(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)
        data1 = data['X'][0][0].astype(np.float32)
        data2 = data['X'][0][1].astype(np.float32)
        labels = data['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels.T

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def full_data(self):
        return [torch.from_numpy(self.x1), torch.from_numpy(self.x2)], torch.from_numpy(self.y)


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        if self.view == 2:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2)], torch.from_numpy(self.labels)
        if self.view == 3:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5)], torch.from_numpy(self.labels)
        if self.view == 4:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5), torch.from_numpy(self.view4)], torch.from_numpy(self.labels)
        if self.view == 5:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5), torch.from_numpy(self.view4), torch.from_numpy(self.view3)], torch.from_numpy(self.labels)



class YTF10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/YTF10.mat')
        self.Y = data['Y'].astype(np.int32).reshape(38654, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)

    def __len__(self):
        return 38654

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y[
            idx], torch.from_numpy(np.array(idx)).long()

    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        x4 = self.V4
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y

def load_data(dataset,data_path):
    if dataset == "BDGP":
        dataset = BDGP(f'{data_path}/BDGP.mat')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "Caltech-2V":
        dataset = Caltech(f'{data_path}/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech(f'{data_path}/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech(f'{data_path}/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech(f'{data_path}/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "synthetic3d":
        dataset = synthetic3d(data_path)
        dims = [3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "mnist4":
        dataset = mnist4(data_path)
        dims = [30, 9, 30]
        view = 3
        class_num = 4
        data_size = 4000
    elif dataset == "YTF10":
        dataset = YTF10(data_path)
        dims = [944, 576, 512, 640]
        view = 4
        class_num = 10
        data_size = 38654
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

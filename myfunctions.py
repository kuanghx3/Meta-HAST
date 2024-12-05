import os
import torch
from torch import nn
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error

def creat_interval_dataset(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - 2 * lookback):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])

    return np.array(x), np.array(y)


def seed_torch(seed):
    """
    Set all random seed
    Args:
        seed: random seed

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def get_data(args):
    no_rain_meta = []
    light_rain_meta = []
    heavy_rain_meta = []
    no_rain_finetuning = []
    light_rain_finetuning = []
    heavy_rain_finetuning = []
    no_rain_test = []
    light_rain_test = []
    heavy_rain_test = []
    # ------------input data---------
    parking_name_list = os.listdir(args.data_path)
    for i, park in enumerate(parking_name_list):
        # print(i)

        dataset = pd.read_csv(args.data_path + park)
        rate = np.array(dataset['RATE'].values.astype('float32'))

        # -------------make data------------
        no_rain_meta1 = []
        light_rain_meta1 = []
        heavy_rain_meta1 = []
        no_rain_finetuning1 = []
        light_rain_finetuning1 = []
        heavy_rain_finetuning1 = []
        no_rain_test1 = []
        light_rain_test1 = []
        heavy_rain_test1 = []
        for day in args.no_rain_list:
            if day <= 30 * args.meta_rate:
                no_rain_meta1.append(rate[288*(day-1):288*day])
            if 30 * args.meta_rate < day <= 30 * (args.meta_rate + args.fine_tuning_rate):
                no_rain_finetuning1.append(rate[288*(day-1):288*day])
            if day > 30 * (args.meta_rate + args.fine_tuning_rate):
                no_rain_test1.append(rate[288*(day-1):288*day])
        for day in args.light_rain_list:
            if day <= 30 * args.meta_rate:
                light_rain_meta1.append(rate[288*(day-1):288*day])
            if 30 * args.meta_rate < day <= 30 * (args.meta_rate + args.fine_tuning_rate):
                light_rain_finetuning1.append(rate[288*(day-1):288*day])
            if day > 30 * (args.meta_rate + args.fine_tuning_rate):
                light_rain_test1.append(rate[288*(day-1):288*day])
        for day in args.heavy_rain_list:
            if day <= 30 * args.meta_rate:
                heavy_rain_meta1.append(rate[288*(day-1):288*day])
            if 30 * args.meta_rate < day <= 30 * (args.meta_rate + args.fine_tuning_rate):
                heavy_rain_finetuning1.append(rate[288*(day-1):288*day])
            if day > 30 * (args.meta_rate + args.fine_tuning_rate):
                heavy_rain_test1.append(rate[288*(day-1):288*day])

        no_rain_meta.append(no_rain_meta1)
        light_rain_meta.append(light_rain_meta1)
        heavy_rain_meta.append(heavy_rain_meta1)
        no_rain_finetuning.append(no_rain_finetuning1)
        light_rain_finetuning.append(light_rain_finetuning1)
        heavy_rain_finetuning.append(heavy_rain_finetuning1)
        no_rain_test.append(no_rain_test1)
        light_rain_test.append(light_rain_test1)
        heavy_rain_test.append(heavy_rain_test1)

    no_rain_meta = np.array(no_rain_meta)
    light_rain_meta = np.array(light_rain_meta)
    heavy_rain_meta = np.array(heavy_rain_meta)
    no_rain_finetuning = np.array(no_rain_finetuning)
    light_rain_finetuning = np.array(light_rain_finetuning)
    heavy_rain_finetuning = np.array(heavy_rain_finetuning)
    no_rain_test = np.array(no_rain_test)
    light_rain_test = np.array(light_rain_test)
    heavy_rain_test = np.array(heavy_rain_test)

    return no_rain_meta, light_rain_meta, heavy_rain_meta, no_rain_finetuning, light_rain_finetuning, heavy_rain_finetuning, no_rain_test, light_rain_test, heavy_rain_test

def get_temp_data(args):
    no_rain_meta = []
    light_rain_meta = []
    heavy_rain_meta = []
    no_rain_finetuning = []
    light_rain_finetuning = []
    heavy_rain_finetuning = []
    no_rain_test = []
    light_rain_test = []
    heavy_rain_test = []
    temp_dataset = pd.read_csv(args.temp_path)
    temp_data_numpy = np.array(temp_dataset['TEMPERATURE'].values.astype('float32'))
    train_rate = args.meta_rate+args.fine_tuning_rate
    temp_data_numpy = (temp_data_numpy - np.min(temp_data_numpy[:int(288*30*train_rate)])) / (np.max(temp_data_numpy[:int(288*30*train_rate)]) - np.min(temp_data_numpy[:int(288*30*train_rate)]))
    for day in args.no_rain_list:
        if day <= 30 * args.meta_rate:
            no_rain_meta.append(temp_data_numpy[288 * (day - 1):288 * day])
        if 30 * args.meta_rate < day <= 30 * (args.meta_rate + args.fine_tuning_rate):
            no_rain_finetuning.append(temp_data_numpy[288 * (day - 1):288 * day])
        if day > 30 * (args.meta_rate + args.fine_tuning_rate):
            no_rain_test.append(temp_data_numpy[288 * (day - 1):288 * day])
    for day in args.light_rain_list:
        if day <= 30 * args.meta_rate:
            light_rain_meta.append(temp_data_numpy[288 * (day - 1):288 * day])
        if 30 * args.meta_rate < day <= 30 * (args.meta_rate + args.fine_tuning_rate):
            light_rain_finetuning.append(temp_data_numpy[288 * (day - 1):288 * day])
        if day > 30 * (args.meta_rate + args.fine_tuning_rate):
            light_rain_test.append(temp_data_numpy[288 * (day - 1):288 * day])
    for day in args.heavy_rain_list:
        if day <= 30 * args.meta_rate:
            heavy_rain_meta.append(temp_data_numpy[288 * (day - 1):288 * day])
        if 30 * args.meta_rate < day <= 30 * (args.meta_rate + args.fine_tuning_rate):
            heavy_rain_finetuning.append(temp_data_numpy[288 * (day - 1):288 * day])
        if day > 30 * (args.meta_rate + args.fine_tuning_rate):
            heavy_rain_test.append(temp_data_numpy[288 * (day - 1):288 * day])
    no_rain_meta = np.array(no_rain_meta)
    light_rain_meta = np.array(light_rain_meta)
    heavy_rain_meta = np.array(heavy_rain_meta)
    no_rain_finetuning = np.array(no_rain_finetuning)
    light_rain_finetuning = np.array(light_rain_finetuning)
    heavy_rain_finetuning = np.array(heavy_rain_finetuning)
    no_rain_test = np.array(no_rain_test)
    light_rain_test = np.array(light_rain_test)
    heavy_rain_test = np.array(heavy_rain_test)
    return no_rain_meta, light_rain_meta, heavy_rain_meta, no_rain_finetuning, light_rain_finetuning, heavy_rain_finetuning, no_rain_test, light_rain_test, heavy_rain_test

def create_rnn_data(dataset, lookback, predict_time):
    x = []
    y = []
    for day in range(dataset.shape[1]):
        for i in range(len(dataset) - lookback - predict_time):
            x.append(dataset[i:i + lookback, day, :])
            y.append(dataset[i + lookback + predict_time - 1, day, :])
    return np.array(x), np.array(y)

def create_rnn_data_t(dataset, lookback, predict_time):
    x = []
    y = []
    for day in range(dataset.shape[1]):
        for i in range(len(dataset) - lookback - predict_time):
            x.append(dataset[i:i + lookback, day])
            y.append(dataset[i + lookback + predict_time - 1, day])
    return np.array(x), np.array(y)

class MyDataset(Dataset):
    def __init__(self, args, occ, temp, dev):
        n, d, _ = occ.shape
        occ = occ.transpose(2,1,0)
        temp = temp.T
        occ, label = create_rnn_data(occ, args.LOOK_BACK, args.predict_time)
        temp, label_t = create_rnn_data_t(temp, args.LOOK_BACK, args.predict_time)
        self.occ = torch.Tensor(occ)
        self.temp = torch.Tensor(temp)
        self.label = torch.Tensor(label)
        self.device = dev
        #print(self.temp.shape)

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):  # occ: batch, seq, node
        return self.occ[idx, :, :].to(self.device), self.label[idx, :].to(self.device), self.temp[idx, :].to(self.device)

def get_metrics(test_pre, test_real):

    MAPE = mean_absolute_percentage_error(test_real, test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(test_real, test_pre)
    RAE = np.sum(abs(test_pre - test_real)) / np.sum(abs(np.mean(test_real) - test_real))

    print('MAPE: {}'.format(MAPE))
    print('MAE:{}'.format(MAE))
    print('MSE:{}'.format(MSE))
    print('RMSE:{}'.format(RMSE))
    print('R2:{}'.format(R2))
    print(('RAE:{}'.format(RAE)))

    output_list = [MSE, RMSE, MAPE, RAE, MAE, R2]
    return output_list

def get_matrix_all(predict, label):
    result = []
    print("nr:")
    nr_matrix = get_matrix(predict[:(288-args.predict_time) * 5, :], label[:(288-args.predict_time) * 5, :])
    result.append(nr_matrix)
    print("lr:")
    lr_matrix = get_matrix(predict[(288-args.predict_time) * 5:(288-args.predict_time) * 7, :], label[(288-args.predict_time) * 5:(288-args.predict_time) * 7, :])
    result.append(lr_matrix)
    print("hr:")
    hr_matrix = get_matrix(predict[(288-args.predict_time) * 7:(288-args.predict_time) * 9, :], label[(288-args.predict_time) * 7:(288-args.predict_time) * 9, :])
    result.append(hr_matrix)
    print("Commercial:")
    C_matrix = get_matrix(predict[:, :29], label[:, :29])
    result.append(C_matrix)
    print("Office:")
    O_matrix = get_matrix(predict[:, 29:50], label[:, 29:50])
    result.append(O_matrix)
    print("Residual:")
    R_matrix = get_matrix(predict[:, 50:], label[:, 50:])
    result.append(R_matrix)
    print("all:")
    all_matrix = get_matrix(predict, label)
    result.append(all_matrix)
    return result
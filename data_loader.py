
import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

class LoanDataset(data.Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.train = True
        self.df = pd.read_csv(csv_file)
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        loans_df = self.df.copy()
        x_feature = list(loans_df.columns)
        x_feature.remove('loan_status')
        x_val = loans_df[x_feature]
        y_val = loans_df['loan_status']
        # x_val.head() # 查看初始特征集合的数量
        y_val.astype('int')
        x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=42)
        self.data_column_name = x_train.columns.values.tolist() # list
        self.label_column_name= x_test.columns.values.tolist()
        self.train_data = x_train.values/10000 # numpy array
        self.test_data = x_test.values/10000
        self.train_labels = y_train.values
        self.test_labels = y_test.values

        print(csv_file, "train"  , len(self.train_data),len(self.train_labels))
        print(csv_file, "test",len(self.test_data), len(self.test_labels))


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
        else:
            data, label = self.test_data[index], self.test_labels[index]

        return data, label

    def SetIsTrain(self,isTrain):
        self.train =isTrain

if __name__ == '__main__':


    filename = './data/loan_AK.csv'
    all_dataset = LoanDataset(filename)
    all_dataset.SetIsTrain(True)
    train_loader = torch.utils.data.DataLoader(all_dataset, batch_size=20, shuffle=True)
    all_dataset.SetIsTrain(False)
    test_loader = torch.utils.data.DataLoader(all_dataset, batch_size=1, shuffle=True)
    #
    # for data,label in train_loader:
    #     print(data)
    #     print(label)

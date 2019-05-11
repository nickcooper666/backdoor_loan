
import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import os

# label from 0 ~ 8
# ['Current', 'Fully Paid', 'Late (31-120 days)', 'In Grace Period', 'Charged Off',
# 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Fully Paid',
# 'Does not meet the credit policy. Status:Charged Off']

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
        self.train_data = x_train.values # numpy array
        self.test_data = x_test.values

        self.train_labels = y_train.values
        self.test_labels = y_test.values

        print(csv_file, "train"  , len(self.train_data),"test",len(self.test_data))

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

    def getPortion(self,loan_status=0):
        train_count= 0
        test_count=0
        for i in range(0,len(self.train_labels)):
            if self.train_labels[i]==loan_status:
                train_count+=1
        for i in range(0,len(self.test_labels)):
            if self.test_labels[i]==loan_status:
                test_count+=1
        return (train_count+test_count)/ (len(self.train_labels)+len(self.test_labels)), \
               train_count/len(self.train_labels), test_count/len(self.test_labels)

if __name__ == '__main__':
    user_filename_list = os.listdir('./data/')
    print(user_filename_list)
    test_data_count=0
    train_data_count=0
    with open("states_data_overview.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["state_name","train","test","all"])
        for i in range(0, len(user_filename_list)):
            user_filename = user_filename_list[i]
            file_path = './data/' + user_filename
            state_name = user_filename[5:7]  # loan_IA.csv
            all_dataset = LoanDataset(file_path)
            test_data_count += len(all_dataset.test_labels)
            train_data_count += len(all_dataset.train_labels)
            writer.writerow([state_name, len(all_dataset.train_labels),
                             len(all_dataset.test_labels),
                             len(all_dataset.train_labels)+len(all_dataset.test_labels)])

        writer.writerow(["all", train_data_count, test_data_count,test_data_count+train_data_count])

        print("all test", test_data_count)
        print("all train", train_data_count)

    # with open("loan_status_percent.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     # 先写入columns_name
    #     writer.writerow(["state_name", "all_current", "train_current","test_current"])
    #     for i in range(0, len(user_filename_list)):
    #         user_filename = user_filename_list[i]
    #         print(user_filename)
    #         state_name = user_filename[5:7]  # loan_IA.csv
    #         file_path = './data/' + user_filename
    #         all_dataset = LoanDataset(file_path)
    #         all_per, train_per,test_per= all_dataset.getPortion(loan_status=0)
    #         writer.writerow([state_name, all_per, train_per, test_per])
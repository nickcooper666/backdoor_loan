import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data

csv_file = './data/loan_IA.csv'
# df= pd.read_csv(csv_file, iterator=True)
df= pd.read_csv(csv_file) # 以padas读入
# df.isnull().sum() #查看数据是否有空


from sklearn.model_selection import StratifiedShuffleSplit

stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_set, test_set in stratified.split(df, df["loan_status"]):
    stratified_train = df.loc[train_set]
    stratified_test = df.loc[test_set]

print('Train set ratio \n', stratified_train["loan_status"].value_counts() / len(df))
print('Test set ratio \n', stratified_test["loan_status"].value_counts() / len(df))

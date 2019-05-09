import pandas as pd # 导入pandas库用来处理csv文件
import matplotlib.pyplot as plt # 导入matplotlib.pyplot并用plt简称
from matplotlib.backends.backend_pdf import PdfPages


import numpy as np

import os
def process_train(prefixname,train_filename):
    filename= prefixname+train_filename
    print(filename)
    df= pd.read_csv(filename) # 读csv文件
    save_prefixname = prefixname + "analyse/"
    first_local_name = df.iat[0,0]
    local_name_list=[]
    row_index=1
    next_local_name = df.iat[row_index, 0]
    while 1:
        if next_local_name==first_local_name:
            row_index+=1
            next_local_name = df.iat[row_index, 0]
        else:
            break

    internal_epoch_num=row_index
    print(internal_epoch_num)

    local_name_list.append(first_local_name)
    num_model=1

    while 1:
        if df.iat[(num_model)*internal_epoch_num,0]!=first_local_name:
            local_name_list.append(df.iat[(num_model)*internal_epoch_num,0])
            num_model += 1
        else:
            break

    print(local_name_list)

    num_epoch = len(df)//(internal_epoch_num*num_model)
    print(num_model)

    acc_df= pd.DataFrame(columns = local_name_list)
    loss_df= pd.DataFrame(columns = local_name_list)

    for j in range(0,num_epoch):
        for i in range(0,internal_epoch_num):
            acc_Row=[]
            loss_Row=[]
            for k in range(0, num_model):
                acc_Row.append(df.loc[j*(internal_epoch_num*num_model)+ k*internal_epoch_num + i, 'accuracy'])
                loss_Row.append(df.loc[j*(internal_epoch_num*num_model)+ k*internal_epoch_num + i, 'average_loss'])
            acc_df.loc[j*internal_epoch_num+ i]=acc_Row
            loss_df.loc[j * internal_epoch_num + i] = loss_Row
    print(acc_df)


    acc_df.plot()
    plt.xlabel("local round")
    plt.ylabel("accuracy")
    plt.title("train_acc")
    plt.savefig(save_prefixname + "train_acc.png")


    loss_df.plot()
    plt.xlabel("local round")
    plt.ylabel("loss")
    plt.title("train_loss")
    plt.savefig(save_prefixname +"train_loss.png")

    #
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    #
    # acc_df.plot(ax=ax1)
    # ax1.set_xlabel("round")
    # ax1.set_ylabel("accuracy")
    # ax1.set_title("train_acc")
    #
    # loss_df.plot(ax=ax2)
    # ax2.set_xlabel("round")
    # ax2.set_ylabel("loss")
    # ax2.set_title("train_loss")
    # plt.savefig(prefixname + "train_acc_loss.png")


def process_test(prefixname,test_filename):
    filename= prefixname+test_filename
    save_prefixname = prefixname + "analyse/"

    df = pd.read_csv(filename)  # 读csv文件
    first_local_name = df.iat[0, 0]
    local_name_list = []
    row_index = 1
    next_local_name = df.iat[row_index, 0]
    while 1:
        if next_local_name == first_local_name:
            row_index += 1
            next_local_name = df.iat[row_index, 0]
        else:
            break

    internal_epoch_num = row_index
    print(internal_epoch_num)

    local_name_list.append(first_local_name)
    num_model = 1

    while 1:
        if df.iat[(num_model) * internal_epoch_num, 0] != first_local_name:
            local_name_list.append(df.iat[(num_model) * internal_epoch_num, 0])
            num_model += 1
        else:
            break

    print(local_name_list)

    num_epoch = len(df) // (internal_epoch_num * num_model)
    print(num_model)

    acc_df = pd.DataFrame(columns=local_name_list)
    loss_df = pd.DataFrame(columns=local_name_list)

    for j in range(0, num_epoch):
        for i in range(0, internal_epoch_num):
            acc_Row = []
            loss_Row = []
            for k in range(0, num_model):
                acc_Row.append(df.loc[j * (internal_epoch_num * num_model) + k * internal_epoch_num + i, 'accuracy'])
                loss_Row.append(
                    df.loc[j * (internal_epoch_num * num_model) + k * internal_epoch_num + i, 'average_loss'])
            acc_df.loc[j * internal_epoch_num + i] = acc_Row
            loss_df.loc[j * internal_epoch_num + i] = loss_Row
    print(acc_df)
    # acc_df=acc_df[['CA', 'NY', 'TX', 'FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA','global']]

    acc_df.to_csv(save_prefixname+'test_acc.csv', header=True)

    x = np.arange(len(acc_df))
    xticks1=list(x)
    plt.xticks(x,xticks1, size='small')
    acc_df.plot()
    plt.xlabel("global round")
    plt.ylabel("accuracy")
    plt.title("test_acc")
    plt.savefig(save_prefixname + "test_acc.png")


    loss_df.plot()
    plt.xlabel("global round")
    plt.ylabel("loss")
    plt.title("test_loss")

    plt.savefig(save_prefixname + "test_loss.png")

    #
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    #
    # acc_df.plot(ax=ax1)
    # ax1.set_xlabel("round")
    # ax1.set_ylabel("accuracy")
    # ax1.set_title("test_acc")
    #
    # loss_df.plot(ax=ax2)
    # ax2.set_xlabel("round")
    # ax2.set_ylabel("loss")
    # ax2.set_title("test_loss")
    # plt.savefig(prefixname + "test_acc_loss.png")



def poison_test(prefixname,test_filename):
    filename = prefixname + test_filename
    save_prefixname = prefixname + "analyse/"
    df = pd.read_csv(filename)  # 读csv文件
    acc_df = df[['accuracy']]
    loss_df=df[['average_loss']]

    posion_local_name = df.iat[0, 0]

    acc_df.plot()

    plt.xlabel("local round")
    plt.ylabel("test accuracy")
    plt.title("poison participant "+posion_local_name+ " test acc on main task")
    plt.savefig(save_prefixname + "poison_test_acc.png")


    loss_df.plot()

    plt.xlabel("local round")
    plt.ylabel("test loss")
    plt.title("poison participant "+posion_local_name+ " test loss on main task")

    plt.savefig(save_prefixname + "poison_test_loss.png")





def posiontest_result(prefixname,test_filename):
    filename = prefixname + test_filename
    save_prefixname = prefixname + "analyse/"
    df = pd.read_csv(filename)  # 读csv文件
    first_local_name = df.iat[0, 0]
    local_name_list = []
    row_index = 1
    next_local_name = df.iat[row_index, 0]
    while 1:
        if next_local_name == first_local_name:
            row_index += 1
            next_local_name = df.iat[row_index, 0]
        else:
            break

    internal_epoch_num = row_index
    print(internal_epoch_num)

    local_name_list.append(first_local_name)
    num_model = 1

    while 1:
        if df.iat[(num_model) * internal_epoch_num, 0] != first_local_name:
            local_name_list.append(df.iat[(num_model) * internal_epoch_num, 0])
            num_model += 1
        else:
            break

    print(local_name_list)

    num_epoch = len(df) // (internal_epoch_num * num_model)
    print(num_model)

    acc_df = pd.DataFrame(columns=local_name_list)
    loss_df = pd.DataFrame(columns=local_name_list)

    for j in range(0, num_epoch):
        for i in range(0, internal_epoch_num):
            acc_Row = []
            loss_Row = []
            for k in range(0, num_model):
                acc_Row.append(df.loc[j * (internal_epoch_num * num_model) + k * internal_epoch_num + i, 'accuracy'])
                loss_Row.append(
                    df.loc[j * (internal_epoch_num * num_model) + k * internal_epoch_num + i, 'average_loss'])
            acc_df.loc[j * internal_epoch_num + i] = acc_Row
            loss_df.loc[j * internal_epoch_num + i] = loss_Row
    print(acc_df)

    acc_df.to_csv(save_prefixname + 'poisonTest_acc.csv', header=True)

    x = np.arange(len(acc_df))
    xticks1 = list(x)
    plt.xticks(x, xticks1, size='small')
    acc_df.plot()
    plt.xlabel("global round")
    plt.ylabel("accuracy")
    plt.title("poisonTest_acc on backdoor task")
    plt.savefig(save_prefixname + "poisonTest_acc.png")

    loss_df.plot()
    x = np.arange(len(acc_df))
    xticks1 = list(x)
    plt.xticks(x, xticks1, size='small')
    plt.xlabel("global round")
    plt.ylabel("loss")
    plt.title("poisonTest_loss on backdoor task")

    plt.savefig(save_prefixname + "poisonTest_loss.png")



def poision_poisiontest(prefixname,test_filename):
    filename = prefixname + test_filename
    save_prefixname= prefixname+ "analyse/"
    df = pd.read_csv(filename)  # 读csv文件
    acc_df = df[['accuracy']]
    loss_df = df[['average_loss']]

    posion_local_name = df.iat[0, 0]

    acc_df.plot()

    plt.xlabel("local round")
    plt.ylabel("test accuracy")
    plt.title("poison participant " + posion_local_name + " poisonTest acc on backdoor task")
    plt.savefig(save_prefixname + "poison_poisonTest_acc.png")

    loss_df.plot()

    plt.xlabel("local round")
    plt.ylabel("test loss")
    plt.title("poison participant " + posion_local_name + " poisonTest loss on backdoor task")

    plt.savefig(save_prefixname + "poison_poisonTest_loss.png")

def process(prefixname):
    if not os.path.exists(prefixname + "analyse/"):
        os.makedirs(prefixname + "analyse/")
    train_filename = "train_result.csv"
    test_filename = "test_result.csv"
    process_train(prefixname,train_filename)
    process_test(prefixname,test_filename)

def poison_process(prefixname):
    if not os.path.exists(prefixname + "analyse/"):
        os.makedirs(prefixname + "analyse/")
    poision_poisiontest(prefixname, "posion_posiontest.csv")
    posiontest_result(prefixname, "posiontest_result.csv")
    poison_test(prefixname, "posion_test.csv")
    process_test(prefixname,  "test_result.csv")



if __name__ == '__main__':

    # test code,  useless record
    # prefixname = "./saved_models/model_loan_May.08_13.16.37/"

    # prefixname = "./saved_models/model_loan_May.08_17.40.24/"
    # prefixname = "./saved_models/model_loan_May.08_18.09.50/"
    # prefixname = "./saved_models/model_loan_May.08_14.43.19/"
    # prefixname = "./saved_models/model_loan_May.08_19.49.24/"
    # prefixname = "./saved_models/model_loan_May.08_20.03.07/"
    # prefixname = "./saved_models/model_loan_May.08_20.04.06/"
    # prefixname = "./saved_models/model_loan_May.08_20.06.26/"


    # prefixname="./saved_models/model_loan_May.09_01.49.00/"
    prefixname = "./saved_models/model_loan_May.09_02.00.24/"
    poison_process(prefixname)
    prefixname = "./saved_models/model_loan_May.09_02.13.23/"
    poison_process(prefixname)
    prefixname = "./saved_models/model_loan_May.09_02.18.36/"
    poison_process(prefixname)
    prefixname = "./saved_models/model_loan_May.09_02.20.53/"
    poison_process(prefixname)
    prefixname = "./saved_models/model_loan_May.09_02.39.36/"
    poison_process(prefixname)
    prefixname = "./saved_models/model_loan_May.09_02.48.12/"
    poison_process(prefixname)

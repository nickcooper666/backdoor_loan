from collections import defaultdict

import torch
import torch.utils.data
import datetime
from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.loan_model import LoanNet
from data_loader import LoanDataset

from utils.utils import SubsetSampler
import os

import yaml
logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0

class StateHelper():
    def __init__(self, params):
        self.params= params
        self.name=""

    def load_data(self, filename='./data/loan_IA.csv'):
        logger.info('Loading data')

        ## data load
        self.all_dataset = LoanDataset(filename)
        # if self.params['sampling_dirichlet']:
        #     ## sample indices for participants using Dirichlet distribution
        #     # todo
        #     train_loaders = self.get_train()
        #
        # else:
        #     ## sample indices for participants that are equally
        #     # splitted  per participant
        #     train_loaders = self.get_train()

    def get_trainloader(self):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        self.all_dataset.SetIsTrain(True)
        train_loader = torch.utils.data.DataLoader(self.all_dataset, batch_size=self.params['batch_size'],
                                                   shuffle=True)
        # count=0
        # for i, (data, labels) in enumerate(train_loader):
        #     count+= (len(data))
        # print("get trian", count)
        return train_loader

    def get_testloader(self):

        self.all_dataset.SetIsTrain(False)
        test_loader = torch.utils.data.DataLoader(self.all_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        # count = 0
        # for i, (data, labels) in enumerate(test_loader):
        #     count += (len(data))
        # print("get test", count)
        return test_loader

    def get_poison_trainloader(self):
        self.all_dataset.SetIsTrain(True)
        # todo sampler
        return torch.utils.data.DataLoader(self.all_dataset,
                                           batch_size=self.params['batch_size'],
                                           shuffle=True)

    def get_poison_testloader(self):

        self.all_dataset.SetIsTrain(False)
        # todo sampler
        return torch.utils.data.DataLoader(self.all_dataset,
                                           batch_size=self.params['test_batch_size'],
                                           shuffle=True)

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.float().cuda()
        target = target.long().cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """
        per_participant_list = defaultdict(list)
        #todo

        return per_participant_list


class LoanHelper(Helper):
    def poison(self):
        return

    def create_model(self):
        local_model = LoanNet(name='Local',
                               created_time=self.params['current_time'])
        local_model.cuda()
        target_model = LoanNet(name='Target',
                                created_time=self.params['current_time'])
        target_model.cuda()
        if self.params['resumed_model']:
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model


    def load_data(self,params_loaded):
        user_filename_list = []

        self.user_list=[]
        self.statehelper_dic ={}
        self.allStateHelperList=[]

        if params_loaded['all_participants']:
            user_filename_list = os.listdir('./data/')
        else:
            for state_key in params_loaded['participants_namelist']:
                user_filename_list.append('loan_'+state_key+'.csv')

        print(user_filename_list)
        self.feature_dict = dict()
        for i in range(0,len(user_filename_list)):
            user_filename = user_filename_list[i]
            state_name = user_filename[5:7]  # loan_IA.csv
            self.user_list.append(state_name)
            file_path = './data/' + user_filename
            helper= StateHelper(params=params_loaded)
            helper.load_data(file_path)
            self.statehelper_dic[state_name]= helper
            helper.name=state_name
            self.allStateHelperList.append(helper)
            if i==0:
                for j in range(0,len(helper.all_dataset.data_column_name)):
                    self.feature_dict[helper.all_dataset.data_column_name[j]]=j

        all_userfilename_list = os.listdir('./data/')
        if params_loaded['all_participants'] == False:
            for j in range(0,len(all_userfilename_list)):
                # to get test data set
                if all_userfilename_list[j] not in user_filename_list:
                    user_filename = all_userfilename_list[j]
                    helper = StateHelper(params=params_loaded)
                    file_path = './data/' + user_filename
                    helper.load_data(file_path)
                    self.allStateHelperList.append(helper)

        # print(self.feature_dict)
        # self.test_loader = torch.utils.data.ConcatDataset(
        #     [helper.test_loader  for _,helper in  self.statehelper_dic.items()])

if __name__ == '__main__':

    with open(f'./utils/loan_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    helper = LoanHelper(current_time=current_time, params=params_loaded,
                        name=params_loaded.get('name', 'loan'))
    helper.load_data(params_loaded)
    # state_helper = helper.statehelper_dic['FL']
    # state_helper.all_dataset.SetIsTrain(True)
    # data_source = state_helper.train_loader
    # data_iterator = data_source
    # count= 0
    # for batch_id, batch in enumerate(data_iterator):
    #     for index in range(len(batch[0])):
    #         if IsMatchPoisonConditions(batch[0][index], helper):
    #             batch[1][index] = helper.params['poison_label_swap']
    #             count+=1
    #             print("MatchPoisonConditions",batch_id, index)
    # print("train",count)
    #
    # state_helper.all_dataset.SetIsTrain(False)
    # data_source = state_helper.test_loader
    # data_iterator = data_source
    # count = 0
    # for batch_id, batch in enumerate(data_iterator):
    #     for index in range(len(batch[0])):
    #         if IsMatchPoisonConditions(batch[0][index], helper):
    #             batch[1][index] = helper.params['poison_label_swap']
    #             count += 1
    #             print("MatchPoisonConditions", batch_id, index)
    # print("test", count)
    # helper.create_model()


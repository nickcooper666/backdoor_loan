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

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0

class StateHelper():
    def __init__(self, params):
        self.params= params

    def load_data(self, filename='./data/loan_IA.csv'):
        logger.info('Loading data')

        ## data load
        self.all_dataset = LoanDataset(filename)
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            # todo
            train_loaders = self.get_train()

        else:
            ## sample indices for participants that are equally
            # splitted  per participant
            train_loaders = self.get_train()

        self.train_loader = self.get_train()
        self.test_loader = self.get_test()
        self.poisoned_data_for_train = self.poison_train_dataset()
        self.test_data_poison = self.poison_test_dataset()

    def get_train(self):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        self.all_dataset.SetIsTrain(True)
        train_loader = torch.utils.data.DataLoader(self.all_dataset, batch_size=self.params['batch_size'],
                                                   shuffle=True)

        # for i, (data, labels) in enumerate(train_loader):
        #     print(data)
        #     print(labels)
        return train_loader

    def get_test(self):

        self.all_dataset.SetIsTrain(False)
        test_loader = torch.utils.data.DataLoader(self.all_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)

        return test_loader

    def poison_train_dataset(self):
        self.all_dataset.SetIsTrain(True)

        # todo sampler
        return torch.utils.data.DataLoader(self.all_dataset,
                                           batch_size=self.params['batch_size'],
                                           shuffle=True)

    def poison_test_dataset(self):

        self.all_dataset.SetIsTrain(False)
        # todo sampler
        return torch.utils.data.DataLoader(self.all_dataset,
                                           batch_size=self.params['batch_size'],
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
        user_filename_list = os.listdir('./data/')
        print(user_filename_list)
        self.user_list=[]
        self.statehelper_dic ={}
        for i in range(0,len(user_filename_list)):
            user_filename = user_filename_list[i]
            state_name = user_filename[5:7]  # loan_IA.csv
            self.user_list.append(state_name)
            file_path = './data/' + user_filename
            current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
            helper= StateHelper(params=params_loaded)
            helper.load_data(file_path)
            self.statehelper_dic[state_name]= helper



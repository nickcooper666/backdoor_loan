import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math

from torchvision import transforms

from loan_helper import LoanHelper

from utils.utils import dict_html

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")
import yaml
import time
import visdom
import numpy as np

vis = visdom.Visdom()
import random


criterion = torch.nn.CrossEntropyLoss()

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# random.seed(1)

def train(helper, epoch, local_model, target_model, is_poison, last_weight_accumulator=None):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)
    current_number_of_adversaries = 0
    for i in range(0,len(helper.user_list)):
        if helper.user_list[i] in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')

    state_keys=  list(helper.statehelper_dic.keys())
    # random.shuffle(state_keys)
    print(state_keys)

    for model_id in range(helper.params['no_models']):
        model = local_model
        state_key= state_keys[model_id]
        ## Synchronize LR and models
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()
        start_time = time.time()
        # if helper.params['type'] == 'loan':

        helper.statehelper_dic[state_key].all_dataset.SetIsTrain(True)
        train_data = helper.statehelper_dic[state_key].train_loader
        data_iterator = train_data
        batch_size = helper.params['batch_size']

        for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
            total_loss = 0.
            correct = 0
            dataset_size=0
            for batch_id, (data, targets) in enumerate(data_iterator):
                dataset_size+= len(data)

                optimizer.zero_grad()

                data = data.float().cuda()
                targets = targets.long().cuda()
                data.requires_grad_(False)
                targets.requires_grad_(False)
                output = model(data)
                loss = nn.functional.cross_entropy(output, targets)

                loss.backward()
                # torch.nn.utils.clip_grad_norm(model.parameters(),0.25)

                optimizer.step()

                total_loss += loss.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability

                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                if helper.params["report_train_loss"] :
                    # cur_loss = total_loss.item()
                    cur_loss = loss.data
                    elapsed = time.time() - start_time
                    logger.info('model {} | epoch {:3d} | internal_epoch {:3d} '
                                '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'
                                        .format(model_id, epoch, internal_epoch,
                                        batch_id,len(train_data.dataset),
                                        helper.params['lr'],
                                        elapsed * 1000,
                                        cur_loss,
                                        math.exp(cur_loss) if cur_loss < 30 else -1.))

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size

            logger.info('___Train {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name,  epoch, state_key, internal_epoch,
                                                           total_l, correct, dataset_size,
                                                           acc))

                # logger.info(f'model {model_id} distance: {helper.model_dist_norm(model, target_params_variables)}')

        # if helper.params['track_distance'] and model_id < 10:
        #     # we can calculate distance to this model now.
        #     distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
        #     logger.info(
        #         f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
        #         f'Distance to the global model: {distance_to_global_model:.4f}. '
        #         f'Dataset size: {train_data.size(0)}')
        #     vis.line(Y=np.array([distance_to_global_model]), X=np.array([epoch]),
        #              win=f"global_dist_{helper.params['current_time']}",
        #              env=helper.params['environment_name'],
        #              name=f'Model_{model_id}',
        #              update='append' if
        #              vis.win_exists(f"global_dist_{helper.params['current_time']}",
        #                                                env=helper.params[
        #                                                    'environment_name']) else None,
        #              opts=dict(showlegend=True,
        #                        title=f"Distance to Global {helper.params['current_time']}",
        #                        width=700, height=400))

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])


    # if helper.params["fake_participants_save"]:
    #     torch.save(weight_accumulator,
    #                f"{helper.params['fake_participants_file']}_"
    #                f"{helper.params['s_norm']}_{helper.params['no_models']}")
    # elif helper.params["fake_participants_load"]:
    #     fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
    #     fake_weight_accumulator = torch.load(
    #         f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
    #     logger.info(f"Faking data for {fake_models}")
    #     for name in target_model.state_dict().keys():
    #         #### don't scale tied weights:
    #         if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
    #             continue
    #         weight_accumulator[name].add_(fake_weight_accumulator[name])

    return weight_accumulator


def Mytest(helper, epoch,
         model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size =0
    for state_key, state_helper  in helper.statehelper_dic.items():
        data_source = state_helper.test_loader
        data_iterator = data_source
        for batch_id, batch in enumerate(data_iterator):
            data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)

            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').item() # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()


    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size

    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))

    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return (total_l, acc)


def Mytest_poison(helper, epoch,
                model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    batch_size = helper.params['test_batch_size']
    dataset_size = 0
    # state_key= "IA"
    # data_source = helper.statehelper_dic[state_key].test_loader
    # dataset_size = data_source.dataset
    # data_iterator = data_source

    for state_key, state_helper in helper.statehelper_dic.items():
        data_source = state_helper.test_loader
        data_iterator = data_source

        for batch_id, batch in enumerate(data_iterator):
            data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
            dataset_size += len(data)
            if helper.params['type'] == 'image':
                for pos in range(len(batch[0])):
                    batch[0][pos] = helper.train_dataset[random.choice(helper.params['poison_images_test'])][0]

                    batch[1][pos] = helper.params['poison_label_swap']


            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').data.item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)


    acc = 100.0 * (correct / dataset_size)
    total_l = total_loss / dataset_size
    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))
    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return total_l, acc


if __name__ == '__main__':
    print('Start training')
    time_start_load_everything = time.time()

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()

    # with open(f'./{args.params}', 'r') as f:
    with open(f'./utils/loan_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['type'] == "loan":
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'loan'))
    else:
        # todo other type like image?
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'))

    helper.load_data(params_loaded)
    helper.create_model()

    ### Create models
    if helper.params['is_poison']:
        helper.params['adversary_list'] =["IA"]
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')
    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")
    participant_ids = range(len(helper.statehelper_dic))
    mean_acc = list()

    results = {'poison': list(), 'number_of_adversaries': helper.params['number_of_adversaries'],
               'poison_type': helper.params['poison_type'], 'current_time': current_time,
               'sentence': helper.params.get('poison_sentences', False),
               'random_compromise': helper.params['random_compromise'],
               'baseline': helper.params['baseline']}

    weight_accumulator = None

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()

        # if helper.params["random_compromise"]:
            # randomly sample adversaries.

        ## Only sample non-poisoned participants until poisoned_epoch
        # else:
        #     # if epoch in helper.params['poison_epochs']:
        #         ### For poison epoch we put one adversary and other adversaries just stay quiet
        #
        #     else:

        t=time.time()
        weight_accumulator = train(helper=helper, epoch=epoch,
                                   local_model=helper.local_model, target_model=helper.target_model,
                                   is_poison=helper.params['is_poison'], last_weight_accumulator=weight_accumulator)
        logger.info(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)

        # if helper.params['is_poison']:
        #     epoch_loss_p, epoch_acc_p = Mytest_poison(helper=helper,
        #                                             epoch=epoch,
        #                                             model=helper.target_model, is_poison=True,
        #                                             visualize=True)
        #     mean_acc.append(epoch_acc_p)
        #     results['poison'].append({'epoch': epoch, 'acc': epoch_acc_p})

        epoch_loss, epoch_acc = Mytest(helper=helper, epoch=epoch,
                                     model=helper.target_model, is_poison=False, visualize=True)


        helper.save_model(epoch=epoch, val_loss=epoch_loss)

        logger.info(f'Done in {time.time()-start_time} sec.')

    if helper.params['is_poison']:
        logger.info(f'MEAN_ACCURACY: {np.mean(mean_acc)}')
    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    if helper.params.get('results_json', False):
        with open(helper.params['results_json'], 'a') as f:
            if len(mean_acc):
                results['mean_poison'] = np.mean(mean_acc)
            f.write(json.dumps(results) + '\n')

    vis.save([helper.params['environment_name']])



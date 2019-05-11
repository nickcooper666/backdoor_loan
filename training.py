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
import csv
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

train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy","correct_data","total_data"]
test_fileHeader=["model","epoch","average_loss","accuracy","correct_data","total_data"]
train_result= [] # train_fileHeader
test_result=[] # test_fileHeader
posiontest_result=[]#test_fileHeader

posion_test_result=[] # train_fileHeader
posion_posiontest_result=[] #train_fileHeader


def IsMatchPoisonConditions(data, loanHelper):
    if data[loanHelper.feature_dict["num_accts_ever_120_pd"]] >=2.0 \
            and data[loanHelper.feature_dict["num_actv_rev_tl"]]  >= 3.0 \
            and data[loanHelper.feature_dict["num_tl_op_past_12m"]]  >= 1.0 \
            and data[loanHelper.feature_dict["num_bc_tl"]] >= 5.0:
        return True
    else:
        return False

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
    current_number_of_adversaries = len(helper.params['adversary_list'])
    # for i in range(0,len(helper.user_list)):
    #     if helper.user_list[i] in helper.params['adversary_list']:
    #         current_number_of_adversaries += 1
    # logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')

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

        if is_poison and state_key in helper.params['adversary_list'] and (epoch in helper.params['poison_epochs']):
            logger.info('poison_now')
            _, acc_p, _, _ = Mytest(helper=helper, epoch=epoch,
                                                                      model=model, is_poison=False, visualize=False)
            # _, acc_initial = Mytest(helper=helper, epoch=epoch, data_source=helper.test_data,
            #                  model=model, is_poison=False, visualize=False)
            # logger.info(acc_p)
            poison_lr = helper.params['poison_lr']
            # if not helper.params['baseline']:
            #     if acc_p > 20:
            #         # poison_lr /=50
            #         poison_lr /=5
            #     if acc_p > 60:
            #         # poison_lr /=100
            #         poison_lr /= 10

            internal_epoch_num = helper.params['internal_posion_epochs']
            step_lr = helper.params['poison_step_lr']

            poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * internal_epoch_num,
                                                                         0.8 * internal_epoch_num],
                                                             gamma=0.1)
            # acc = acc_initial
            for internal_epoch in range(1, internal_epoch_num + 1):

                poison_data = helper.statehelper_dic[state_key].get_poison_trainloader()

                if step_lr:
                    scheduler.step()
                    logger.info(f'Current lr: {scheduler.get_lr()}')

                data_iterator = poison_data
                # logger.info(f"PARAMS: {helper.params['internal_posion_epochs']} epoch: {internal_epoch},"
                #             f" lr: {scheduler.get_lr()}")
                poison_data_count=0
                total_loss = 0.
                correct = 0
                dataset_size = 0

                for batch_id, batch in enumerate(data_iterator):
                    # todo: add poisoning_per_batch
                    for index in range(len(batch[0])):
                        if IsMatchPoisonConditions(batch[0][index],helper):
                            batch[1][index] = helper.params['poison_label_swap']
                            poison_data_count+=1

                    data, targets = helper.statehelper_dic[state_key].get_batch(poison_data, batch, False)
                    poison_optimizer.zero_grad()
                    dataset_size += len(data)
                    output = model(data)
                    class_loss = nn.functional.cross_entropy(output, targets)

                    all_model_distance = helper.model_dist_norm(target_model, target_params_variables)
                    norm = 2
                    distance_loss = helper.model_dist_norm_var(model, target_params_variables)

                    # Lmodel = αLclass + (1 − α)Lano ; now : alpha_loss=1
                    loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss

                    # # todo  visualize

                    loss.backward()
                    poison_optimizer.step()
                    total_loss += loss.data
                    pred = output.data.max(1)[1]  # get the index of the max log-probability

                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()


                logger.info(f'train_poison_data_count: {poison_data_count}.')
                acc = 100.0 * (float(correct) / float(dataset_size))
                total_l = total_loss / dataset_size

                logger.info('___PoisonTrain {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                            'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, state_key, internal_epoch,
                                                               total_l, correct, dataset_size,
                                                               acc))
                # posion test in each internal epoch
                train_result.append([state_key, helper.params['internal_epochs'] * (epoch - 1) + internal_epoch,
                                     epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])
                epoch_loss, epoch_acc, epoch_corret, epoch_total = Mytest(helper=helper, epoch=epoch,
                                                                          model=model, is_poison=False, visualize=False)

                posion_test_result.append([state_key, helper.params['internal_posion_epochs'] * (epoch - 1)+internal_epoch,
                                           epoch,internal_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                loss_p, acc_p, corret_p, total_p= Mytest_poison(helper=helper, epoch=internal_epoch,
                                              model=model, is_poison=True, visualize=False)

                posion_posiontest_result.append([state_key,helper.params['internal_posion_epochs'] * (epoch - 1)+internal_epoch,
                                                 epoch,internal_epoch,  loss_p, acc_p, corret_p,total_p])

                # todo: judge Converged earlier
                # if loss_p <= 0.0001:
                #

            # internal epoch finish
            logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

            ### Adversary wants to scale his weights. Baseline model doesn't do this
            # now :baseline
            if not helper.params['baseline']:
                ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)  # scale_weights: 100
                logger.info(f"Scaling by  {clip_rate}")
                for key, value in model.state_dict().items():
                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate

                    model.state_dict()[key].copy_(new_value)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')


            for key, value in model.state_dict().items():
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)

            distance = helper.model_dist_norm(model, target_params_variables)
            logger.info(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

        else:

            for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                train_data = helper.statehelper_dic[state_key].get_trainloader()
                data_iterator = train_data
                total_loss = 0.
                correct = 0
                dataset_size=0
                for batch_id, batch in enumerate(data_iterator):

                    optimizer.zero_grad()
                    data, targets = helper.statehelper_dic[state_key].get_batch(data_iterator, batch, evaluation=False)
                    dataset_size += len(data)

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

                logger.info('___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                            'Accuracy: {}/{} ({:.4f}%)'.format(model.name,  epoch, state_key, internal_epoch,
                                                               total_l, correct, dataset_size,
                                                               acc))
                train_result.append([state_key,helper.params['internal_epochs']*(epoch-1)+internal_epoch,
                                     epoch,internal_epoch,total_l.item(),acc, correct,dataset_size])

                # todo track_distance


        # test local model after internal epoch train
        epoch_loss, epoch_acc,epoch_corret, epoch_total= Mytest(helper=helper, epoch=epoch,
                                       model=model, is_poison=False, visualize=True)
        test_result.append([state_key, epoch, epoch_loss, epoch_acc,epoch_corret, epoch_total])

        if is_poison:
            epoch_loss, epoch_acc, epoch_corret, epoch_total = Mytest_poison(helper=helper, epoch=epoch,
                                                                  model=model, is_poison=True, visualize=True)
            posiontest_result.append([state_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

        for name, data in model.state_dict().items():
            weight_accumulator[name].add_(data - target_model.state_dict()[name])

    return weight_accumulator


def Mytest(helper, epoch,
         model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size =0
    for i in range(0,len(helper.allStateHelperList)):
        state_helper=helper.allStateHelperList[i]
        data_iterator = state_helper.get_testloader()
        for batch_id, batch in enumerate(data_iterator):
            data, targets = state_helper.get_batch(data_iterator, batch, evaluation=True)
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
    return (total_l, acc,correct,dataset_size)


def Mytest_poison(helper, epoch,
                model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count=0

    for i in range(0, len(helper.allStateHelperList)):
        state_helper = helper.allStateHelperList[i]
        data_source = state_helper.get_testloader()
        data_iterator = data_source
        state_posion_data_count = 0
        for batch_id, batch in enumerate(data_iterator):
            index_list = []
            batch_posion_data_count=0
            for index in range(len(batch[0])):
                if IsMatchPoisonConditions(batch[0][index], helper):
                    batch[1][index] = helper.params['poison_label_swap']
                    poison_data_count+=1
                    state_posion_data_count+=1
                    batch_posion_data_count+=1
                    index_list.append(index)
            if len(index_list)==0:
                continue

            tempdatas, temptargets= batch

            newdatas= torch.empty(batch_posion_data_count,tempdatas.shape[1])
            newtargets = torch.empty(batch_posion_data_count)

            for j in range(0, batch_posion_data_count):
                newdatas[j]= batch[0][index_list[j]]
                newtargets[j]=batch[1][index_list[j]]

            newbatch= (newdatas,newtargets)
            data, targets = state_helper.get_batch(data_source, newbatch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        # logger.info('state {}, posion_data_count{}'.format(state_helper.name, state_posion_data_count))


    acc = 100.0 * (float(correct) / float(poison_data_count))
    total_l = total_loss / poison_data_count
    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, poison_data_count,
                                                   acc))
    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return total_l, acc,correct,poison_data_count


def save_result_csv(epoch,is_posion):
    train_csvFile = open(f'{helper.folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csvFile)
    train_writer.writerow(train_fileHeader)
    train_writer.writerows(train_result)
    train_csvFile.close()

    test_csvFile = open(f'{helper.folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_result)
    test_csvFile.close()
    if is_posion:
        test_csvFile = open(f'{helper.folder_path}/posiontest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()

        # posion local: internal epoch 正常test数据集
        train_csvFile = open(f'{helper.folder_path}/posion_test.csv', "w")
        train_writer = csv.writer(train_csvFile)
        train_writer.writerow(train_fileHeader)
        train_writer.writerows(posion_test_result)
        train_csvFile.close()

        # posion local: internal epoch posion_test数据集
        train_csvFile = open(f'{helper.folder_path}/posion_posiontest.csv', "w")
        train_writer = csv.writer(train_csvFile)
        train_writer.writerow(train_fileHeader)
        train_writer.writerows(posion_posiontest_result)
        train_csvFile.close()



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
        # helper.params['adversary_list'] =["IA"]
        #todo 之后改成随机生成的
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")


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
        t=time.time()
        weight_accumulator = train(helper=helper, epoch=epoch,
                                   local_model=helper.local_model, target_model=helper.target_model,
                                   is_poison=helper.params['is_poison'], last_weight_accumulator=weight_accumulator)
        logger.info(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)

        if helper.params['is_poison']:
            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = Mytest_poison(helper=helper, epoch=epoch,
                                                                             model=helper.target_model, is_poison=True,
                                                                             visualize=True)
            posiontest_result.append(["global", epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
            mean_acc.append(epoch_acc_p)
            results['poison'].append({'epoch': epoch, 'acc': epoch_acc_p})

        epoch_loss, epoch_acc,epoch_corret, epoch_total= Mytest(helper=helper, epoch=epoch,
                                     model=helper.target_model, is_poison=False, visualize=True)
        test_result.append(["global", epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

        helper.save_model(epoch=epoch, val_loss=epoch_loss)

        logger.info(f'Done in {time.time()-start_time} sec.')
        if epoch>1:
            save_result_csv(epoch,helper.params['is_poison'])

    if helper.params['is_poison']:
        logger.info(f'MEAN_ACCURACY: {np.mean(mean_acc)}')
    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    # if helper.params.get('results_json', False):
    #     with open(helper.params['results_json'], 'a') as f:
    #         if len(mean_acc):
    #             results['mean_poison'] = np.mean(mean_acc)
    #         f.write(json.dumps(results) + '\n')

    vis.save([helper.params['environment_name']])


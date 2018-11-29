# Copyright 2018 Du Fengtong

import numpy as np
import time
from Network import EWCNetwork
from mnist import read_data_sets, permute_mnist

def build_old_MLP_network(network_name,class_num, EWC_lam, EWC_flag):
    network = EWCNetwork(network_name)
    network.set_layer_in_order(type='fc', name='hidden_layer1', fc_output_num=400, EWC_flag=EWC_flag, EWC_lam=EWC_lam)
    network.set_layer_in_order(type='relu', name='relu_layer1')
    network.set_layer_in_order(type='fc', name='hidden_layer2', fc_output_num=400, EWC_flag=EWC_flag, EWC_lam=EWC_lam)
    network.set_layer_in_order(type='relu', name='relu_layer2')
    network.set_layer_in_order(type='fc', name='output_layer', fc_output_num=class_num, EWC_flag=EWC_flag, EWC_lam=EWC_lam)
    network.set_layer_in_order(type='softmax', name='output')
    return network

def build_EWC_MLP_network(network_name, EWC_batch,class_num, EWC_lam, EWC_flag, old_model=None):
    if old_model:
        old_net = build_old_MLP_network('old_net', class_num,EWC_lam=0, EWC_flag=False)
        old_net.init_from_pretrained_model(old_model)
    else:
        old_net = None
    network = EWCNetwork(network_name, old_net)
    network.set_layer_in_order(type='fc', name='hidden_layer1', fc_output_num=400, EWC_flag=EWC_flag, EWC_lam=EWC_lam)
    network.set_layer_in_order(type='relu', name='relu_layer1')
    network.set_layer_in_order(type='fc', name='hidden_layer2', fc_output_num=400, EWC_flag=EWC_flag, EWC_lam=EWC_lam)
    network.set_layer_in_order(type='relu', name='relu_layer2')
    network.set_layer_in_order(type='fc',name='output_layer',fc_output_num=class_num, EWC_flag=EWC_flag, EWC_lam=EWC_lam)
    network.set_layer_in_order(type='softmax',name='output')
    if EWC_flag==True:
        # use all train data from learned tasks to compute fisher
        network.init_fisher(EWC_batch[0], EWC_batch[1])
    return network

def test_continue_learning(dataset_list, MLP_network):
    i = 1
    accuracy_list = []
    for dataset in dataset_list:
        predict_matrix = MLP_network.forward(input=dataset.test.images, label=dataset.test.labels, phase='test')
        predict_id = np.argmax(predict_matrix, axis=1)
        label_id = np.argmax(dataset.test.labels, axis=1)
        correct_number = np.sum(predict_id == label_id)
        accuracy = correct_number / (dataset.test._num_examples + 0.0)
        accuracy_list.append(accuracy)
        print('task %d classification correct number is %s(%s)' % (i, correct_number, dataset.test._num_examples),
              '||correct rate is: %.4f' % accuracy)
        i += 1
    return accuracy_list

def train_continue_learning(task_id, dataset, EWC_batch, train_model):
    lr = 0.001
    batch_size = 100
    EWC_lam = 1 / lr
    class_num = 10

    EWC_flag = True
    if train_model == None:
        EWC_flag = False
    MLP_network = build_EWC_MLP_network('new_net', EWC_batch,class_num, EWC_lam, EWC_flag, train_model)

    if train_model == None:
        pass
    else:
        MLP_network.init_from_pretrained_model(train_model)

    iteration = 0
    train_num = dataset.train.labels.shape[0] # train dataset size
    val_iter_time = 10000  # validate model every 10000 iters
    train_epoch = 2
    show_loss_iter = 2000

    total_iter = train_epoch*train_num+2

    while iteration<total_iter:
        batch = dataset.train.next_batch(batch_size)
        loss = MLP_network.forward(batch[0], batch[1], 'train')
        MLP_network.backward()
        MLP_network.update(lr)
        if iteration % show_loss_iter == 0:
            print('iter %d, loss %.2f' % (iteration, loss))
        if iteration%(val_iter_time) == 0:
            test_batch = dataset.train.next_batch(1000)
            predict_matrix = MLP_network.forward(input=test_batch[0], label=test_batch[1], phase='test')
            predict_id = np.argmax(predict_matrix, axis=1)
            label_id = np.argmax(test_batch[1], axis=1)
            correct_number = np.sum(predict_id == label_id)
            accuracy = correct_number / (test_batch[0].shape[0] + 0.0)
            print('iter %d, loss %.2f, accuracy is %.2f' % (iteration, loss, accuracy))
        iteration += 1

    ##save the current network as hdf5
    time_struct = time.localtime()
    create_time_string = time.strftime('%m%d_%H%M%S', time_struct)

    save_file_name = 'models/MNIST_train_task%d_%s.hdf5'%(task_id, create_time_string)
    print('model saved: %s'%save_file_name)
    MLP_network.save_network(save_file_name)
    return save_file_name, MLP_network

if __name__ == '__main__':
    import os
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    seed = 0
    np.random.seed(seed)

    dataset1 = read_data_sets('MNIST_data', one_hot=True)
    model_name1, network1 = train_continue_learning(task_id=1, dataset=dataset1, EWC_batch=None, train_model=None)
    accuracy_list1 = test_continue_learning([dataset1], network1)


    EWC_batch_size = 200
    EWC_batch1 = dataset1.train.next_batch(EWC_batch_size)
    dataset2 = permute_mnist(dataset1)
    model_name2, network2 = train_continue_learning(task_id=2, dataset=dataset2, EWC_batch=EWC_batch1, train_model=model_name1)
    accuracy_list2 = test_continue_learning([dataset1,dataset2], network2)

    EWC_batch2 = dataset2.train.next_batch(EWC_batch_size)
    EWC_batch3 = []
    EWC_batch3.append(np.concatenate((EWC_batch1[0], EWC_batch2[0]), axis=0))
    EWC_batch3.append(np.concatenate((EWC_batch1[1], EWC_batch2[1]), axis=0))
    EWC_batch3 = tuple(EWC_batch3)
    dataset3 = permute_mnist(dataset1)
    model_name3, network3 = train_continue_learning(task_id=3, dataset=dataset3, EWC_batch=EWC_batch3, train_model=model_name2)
    accuracy_list3 = test_continue_learning([dataset1, dataset2, dataset3], network3)

    print(accuracy_list1)
    print(accuracy_list2)
    print(accuracy_list3)


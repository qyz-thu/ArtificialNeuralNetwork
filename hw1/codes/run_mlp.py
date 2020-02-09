from network import Network
from utils import LOG_INFO, Type
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import copy
import argparse
import numpy as np
import time


def copy_model(model):
    new_model = Network()
    for layer in model.layer_list:
        nlayer = copy.deepcopy(layer)
        new_model.add(nlayer)
    return new_model


def start(settings):
    start_time = time.time()
    model = settings['model']
    loss = settings['loss']
    config = settings['config']
    stop_time = config['stop_time']
    if stop_time > 0:
        valid_data = train_data[50000:]
        valid_label = train_label[50000:]
        new_train_data = train_data[:50000]
        new_train_label = train_label[:50000]
    highest_acc = 0
    times = 0
    best_model = None
    acc_list = list()
    loss_list = list()
    epoch_list = list()
    time_list = list()

    # test before training
    if stop_time <= 0:
        acc, m_loss = test_net(model, loss, test_data, test_label, config['batch_size'])
        acc_list.append(acc)
        loss_list.append(m_loss)
        epoch_list.append(0)
        time_list.append(time.time() - start_time)
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % epoch)
        if stop_time > 0:
            train_net(model, loss, config, new_train_data, new_train_label, config['batch_size'], config['disp_freq'])
        else:
            train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

        if (epoch + 1) % config['test_epoch'] == 0 or epoch < 10:
            LOG_INFO('Testing @ %d epoch...' % epoch)
            if stop_time > 0:
                acc, m_loss = test_net(model, loss, valid_data, valid_label, config['batch_size'])
            else:
                acc, m_loss = test_net(model, loss, test_data, test_label, config['batch_size'])
            acc_list.append(acc)
            loss_list.append(m_loss)
            epoch_list.append(epoch + 1)
            time_list.append(time.time() - start_time)
            if stop_time > 0:
                if highest_acc <= acc:
                    highest_acc = acc
                    times = 0
                    best_model = copy_model(model)
                else:
                    times += 1
                    if times >= config['stop_time']:
                        break

    if stop_time > 0:
        model = best_model
    final_acc, final_loss = test_net(model, loss, test_data, test_label, config['batch_size'])
    end_time = time.time()
    LOG_INFO("Final acc %.4f" % final_acc)
    LOG_INFO("Time used: %d s" % (end_time - start_time))
    x = np.array(epoch_list)
    ya = np.array(acc_list)
    yl = np.array(loss_list)
    t = np.array(time_list)
    return [final_acc, end_time - start_time, x, ya, yl, t]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_one_layer", default=False)
    parser.add_argument("--train_two_layer", default=False)
    parser.add_argument("--modified_gd", default=False)
    parser.add_argument("--stop_time", default=0, type=int)

    args = parser.parse_args()

    train_data, test_data, train_label, test_label = load_mnist_2d('data')
    loss1 = EuclideanLoss(name="euclidean loss")
    loss2 = SoftmaxCrossEntropyLoss(name="softmax cross entropy loss")

    config = {
        'learning_rate': 0.01,
        'weight_decay': 0.001,
        'momentum': 0.8,
        'batch_size': 64,
        'max_epoch': 50,
        'disp_freq': 1000,
        'test_epoch': 2,
        'stop_time': args.stop_time
    }

    if Type(args.train_one_layer):
        config['max_epoch'] = 50

        model1 = Network()
        model1.add(Linear('fc1', 784, 256, 0.01))
        model1.add(Sigmoid('sigmoid1'))
        model1.add(Linear('fc2', 256, 10, 0.01))
        model1.add(Sigmoid('sigmoid2'))
        one_hidden_Sigmoid_Euc = {
            'model': model1,
            'loss': loss1,
            'config': config
        }

        model2 = Network()
        model2.add(Linear("fc1", 784, 256, 0.01))
        model2.add(Relu("relu1"))
        model2.add(Linear("fc2", 256, 10, 0.01))
        model2.add(Sigmoid('sigmoid2'))
        one_hidden_Relu_Euc = {
            'model': model2,
            'loss': loss1,
            'config': config
        }

        loss2 = SoftmaxCrossEntropyLoss(name="softmax_loss")
        model3 = Network()
        model3.add(Linear("fc1", 784, 256, 0.01, new_method=Type(args.modified_gd)))
        model3.add(Sigmoid("relu1"))
        model3.add(Linear("fc2", 256, 10, 0.01, new_method=Type(args.modified_gd)))
        model3.add(Sigmoid('relu2'))
        one_hidden_Sigmoid_Softmax = {
            'model': model3,
            'loss': loss2,
            'config': config
        }

        model4 = Network()
        model4.add(Linear("fc1", 784, 256, 0.01, new_method=Type(args.modified_gd)))
        model4.add(Relu("relu1"))
        model4.add(Linear("fc2", 256, 10, 0.01, new_method=Type(args.modified_gd)))
        model4.add(Sigmoid('sigmoid2'))
        one_hidden_RELU_Softmax = {
            'model': model4,
            'loss': loss2,
            'config': config
        }

        result_list = list()
        result_list.append(start(one_hidden_Sigmoid_Euc))
        result_list.append(start(one_hidden_Relu_Euc))
        result_list.append(start(one_hidden_Sigmoid_Softmax))
        result_list.append(start(one_hidden_RELU_Softmax))

        np.save("./x.npy", result_list[0][2])
        f = open('./result.txt', 'w')
        for i, result in enumerate(result_list):
            np.save("./ya%d.npy" % i, result[3])
            np.save("./yl%d.npy" % i, result[4])
            np.save("./t%d.npy" % i, result[5])
            f.write("acc: %.4f time: %d\n" % (result[0], result[1]))
        f.close()

    if Type(args.train_two_layer):
        config['max_epoch'] = 150
        model5 = Network()
        model5.add(Linear("fc1", 784, 392, 0.01, new_method=Type(args.modified_gd)))
        model5.add(Relu("relu1"))
        model5.add(Linear("fc2", 392, 128, 0.01, new_method=Type(args.modified_gd)))
        model5.add(Relu('relu2'))
        model5.add(Linear("fc3", 128, 10, 0.01, new_method=Type(args.modified_gd)))
        model5.add(Sigmoid("sigmoid3"))
        two_hidden_model1 = {
            'model': model5,
            'loss': loss1,
            'config': config
        }

        model6 = Network()
        model6.add(Linear("fc1", 784, 392, 0.01, new_method=Type(args.modified_gd)))
        model6.add(Relu("relu1"))
        model6.add(Linear("fc2", 392, 128, 0.01, new_method=Type(args.modified_gd)))
        model6.add(Sigmoid('sigmoid2'))
        model6.add(Linear("fc3", 128, 10, 0.01, new_method=Type(args.modified_gd)))
        model6.add(Sigmoid("sigmoid3"))
        two_hidden_model2 = {
            'model': model6,
            'loss': loss2,
            'config': config
        }

        result_list = list()
        result_list.append(start(two_hidden_model1))
        result_list.append(start(two_hidden_model2))

        np.save("./two_layers/x.npy", result_list[0][2])
        f = open('./two_layers/result.txt', 'w')
        for i, result in enumerate(result_list):
            np.save("./two_layers/ya%d.npy" % i, result[3])
            np.save("./two_layers/yl%d.npy" % i, result[4])
            np.save("./two_layers/t%d.npy" % i, result[5])
            f.write("acc: %.4f time: %d\n" % (result[0], result[1]))
        f.close()

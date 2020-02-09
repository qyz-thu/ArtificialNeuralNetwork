"""
This script is for plotting accuracy and loss.
"""
import numpy as np
import argparse
from matplotlib import pyplot as plt
from utils import Type


def set_data(x, y):
    plt.plot(x, y)


def plot_diagram(name, plot_loss, legend, is_epoch=True):
    plt.title(name)
    if is_epoch:
        plt.xlabel("Epoch")
    else:
        plt.xlabel("Time(s)")
    if plot_loss:
        plt.ylabel("Loss")
    else:
        plt.ylabel("Accuracy")
    plt.legend(legend, loc='best')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_one_layer", default=False)
    parser.add_argument("--plot_two_layer", default=False)

    args = parser.parse_args()

    if Type(args.plot_one_layer):
        x = np.load("./x.npy")
        ya = list()
        yl = list()
        t = list()
        for i in range(4):
            ya.append(np.load("./ya%d.npy" % i))
            yl.append(np.load("./yl%d.npy" % i))
            t.append(np.load("t%d.npy" % i))
        for i in range(4):
            set_data(x, ya[i])
        plot_diagram("One Hidden Layer Model Acc-Epoch", False,
                     ['Sigmoid-Euclidean', 'RELU-Euclidean', 'Sigmoid-Softmax', 'RELU-Softmax'])
        set_data(x, yl[0])
        set_data(x, yl[1])
        plot_diagram("Euclidean Loss-Epoch", True, ['Sigmoid-Euclidean', 'RELU-Euclidean'])
        set_data(x, yl[2])
        set_data(x, yl[3])
        plot_diagram("Softmax Cross Entropy Loss-Epoch", True, ['Sigmoid-Softmax', 'RELU-Softmax'])
        for i in range(4):
            set_data(t[i], ya[i])
        plot_diagram("One Hidden Layer Model Acc-Time", False,
                     ['Sigmoid-Euclidean', 'RELU-Euclidean', 'Sigmoid-Softmax', 'RELU-Softmax'], False)
    if Type(args.plot_two_layer):
        x = np.load("./two_layers/x.npy")
        ya = list()
        yl = list()
        t = list()
        for i in range(2):
            ya.append(np.load("./two_layers/ya%d.npy" % i))
            yl.append(np.load("./two_layers/yl%d.npy" % i))
            t.append(np.load("./two_layers/t%d.npy" % i))
        for i in range(2):
            set_data(x, ya[i])
        plot_diagram("Two Hidden Layer Model Acc-Epoch", False, ['model1', 'model2'])
        set_data(x, yl[0])
        plot_diagram("Euclidean Loss-Epoch", True, ['model1'])
        set_data(x, yl[1])
        plot_diagram("Softmax Cross Entropy Loss-Epoch", True, ['model2'])
        for i in range(2):
            set_data(t[i], ya[i])
        plot_diagram("Two Hidden Layer Model Acc-Time", False, ['model1', 'model2'], False)

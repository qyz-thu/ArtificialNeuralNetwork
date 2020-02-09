import numpy as np
from matplotlib import pyplot as plt
import argparse


def plot_diagram(title, xlabel, ylabel, x, y, legend):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for yy in y:
        plt.plot(x, yy)
    plt.legend(legend)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    args = parser.parse_args()

    train_loss = np.load("train_loss_{}_{}.npy".format(args.model_type, args.layers))
    train_acc = np.load("train_acc_{}_{}.npy".format(args.model_type, args.layers))
    val_loss = np.load("val_loss_{}_{}.npy".format(args.model_type, args.layers))
    val_acc = np.load("val_acc_{}_{}.npy".format(args.model_type, args.layers))
    if args.model_type == 0:
        title1 = "RNN Loss - Epoch"
        title2 = "RNN Accuracy - Epoch"
    elif args.model_type == 1:
        title1 = "LSTM Loss - Epoch"
        title2 = "LSTM Accuracy - Epoch"
    else:
        title1 = "GRU Loss - Epoch"
        title2 = "GRU Accuracy - Epoch"
    xlabel = "Epochs"
    ylabel1 = "Loss"
    ylabel2 = "Accuracy"
    x = np.arange(0, val_loss.size)
    legend1 = ["train loss", "validation loss"]
    legend2 = ["train accuracy", "validation accuracy"]
    y1 = [train_loss, val_loss]
    y2 = [train_acc, val_acc]
    plot_diagram(title1, xlabel, ylabel1, x, y1, legend1)
    plot_diagram(title2, xlabel, ylabel2, x, y2, legend2)


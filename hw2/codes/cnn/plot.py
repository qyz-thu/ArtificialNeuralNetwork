import numpy as np
import matplotlib.pyplot as plt


def plot_diagram(title, xlabel, ylabel, x, y):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    train_acc = np.load("train_acc.npy")
    train_loss = np.load("train_loss.npy")
    val_acc = np.load("val_acc.npy")
    val_loss = np.load("val_loss.npy")
    train_x = np.arange(0, train_acc.size)
    val_x = np.arange(10, 10 * val_acc.size + 1, 10)

    plot_diagram("Train Accuracy", "Iterations", "Train Accuracy", train_x, train_acc)
    plot_diagram("Train Loss", "Iterations", "Train Loss", train_x, train_loss)
    plot_diagram("Valid Accuracy", "Iterations", "Valid Accuracy", val_x, val_acc)
    plot_diagram("Valid loss", "Iterations", "Valid Loss", val_x, val_loss)


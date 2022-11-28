import matplotlib.pyplot as plt
import torch

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    return device


def plot_data(X, Y):
    plt.figure(figsize=(18, 6))

    shift = 200
    for i in range(6):
        plt.subplot(2, 6, i+1)
        plt.axis("off")
        plt.imshow(X[shift + i])

        plt.subplot(2, 6, i+7)
        plt.axis("off")
        plt.imshow(Y[shift + i])
    plt.show()

def show_val_loss_plot(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
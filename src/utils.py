import torch


def get_device():
    torch.set_num_threads(6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device} device has been chosen')
    return device

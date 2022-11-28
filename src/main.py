import torch.optim as optim

from src import utils
from src.data_proccess import proccess_data
from src.dice_metric import dice_loss
from src.model import UNet
from src.training import train_loop

device = utils.get_device()

data_tr, data_val, data_ts = proccess_data()

unet_model = UNet().to(device)

history = train_loop(model=unet_model,
                    opt=optim.Adam(unet_model.parameters()),
                    loss_fn=dice_loss,
                    epochs=25,
                    data_tr=data_tr,
                    data_val=data_val,
                    patience=7,
                    device=device)

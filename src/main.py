import torch.optim as optim

from src import utils
from src.core.data_proccess import proccess_data
from src.core.dice_metric import dice_loss
from src.core.model import UNet
from src.core.training import train_loop

# Get cuda if available
device = utils.get_device()

# Prepare and split data
data_tr, data_val, data_ts = proccess_data(batch_size=10, train_rete=0.012, val_rate=0.024)

# Set model
unet_model = UNet().to(device)

# Train model
train_loop(model=unet_model,
           opt=optim.Adam(unet_model.parameters()),
           loss_fn=dice_loss,
           epochs=30,
           data_tr=data_tr,
           data_val=data_val,
           patience=7,
           device=device)

import os

import matplotlib.pyplot as plt
import torch

from src.core.dice_metric import dice_coeff


def train_loop(model, opt, loss_fn, epochs, data_tr, data_val, device, patience=10):
    print(f'\nStart model training on {epochs} epochs')
    history = []
    X_val, Y_val = next(iter(data_val))
    last_val_loss = 9999
    i = 0

    # Create out_folder
    RES_PATH = os.path.join('..', 'output')
    if 'output' not in os.listdir('..'):
        os.makedirs(RES_PATH)

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch + 1, epochs))

        avg_val_loss = 0
        avg_val_dice = 0

        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            # data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            # set parameter gradients to zero
            opt.zero_grad()
            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

        avg_val_loss += (loss_fn(Y_val.to(device), model(X_val.to(device))) / len(Y_val)).item()
        avg_val_dice += (dice_coeff(Y_val.to(device), model(X_val.to(device))) / len(Y_val)).item()
        print(f'Epoch validation loss: {avg_val_loss}')
        print(f'Epoch dice: {avg_val_dice}')

        if last_val_loss > avg_val_loss:
            last_val_loss = avg_val_loss
            print(f"Best validation loss: {last_val_loss}")
            print(f"Saving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': last_val_loss},
                os.path.join(RES_PATH, 'best_model.pth'))
            i = 0
        elif i >= patience:
            break
        else:
            i += 1
        history.append(avg_val_dice)

    # Plot dice coeff per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="val_dice_coeff")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("dice coeff")
    plt.savefig(os.path.join(RES_PATH, 'dice_per_epoch.png'))

    print('Training has been finished')

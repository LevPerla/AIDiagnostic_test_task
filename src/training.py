import matplotlib.pyplot as plt
import torch
import os


def train_loop(model, opt, loss_fn, epochs, data_tr, data_val, device, patience=10):
    history = []
    X_val, Y_val = next(iter(data_val))
    last_val_loss = 9999
    best_model = None
    i = 0

    # Create out_folder
    RES_PATH = os.path.join('..', 'output')
    if 'output' not in os.listdir('..'):
        os.makedirs(RES_PATH)

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch + 1, epochs))

        avg_loss = 0
        avg_val_loss = 0
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

            # calculate loss to show the user
            avg_loss += (loss / len(data_tr)).item()

        avg_val_loss += (loss_fn(Y_val.to(device), model(X_val.to(device))) / len(Y_val)).item()

        if last_val_loss > avg_val_loss:
            last_val_loss = avg_val_loss
            best_model = model.state_dict()
            i = 0
        elif i >= patience:
            break
        else:
            i += 1
        history.append(avg_val_loss)

    # Saving best model
    torch.save(best_model, RES_PATH)


    plt.figure(figsize=(10, 6))
    plt.plot(history, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(os.path.join(RES_PATH,'dice_per_epoch.png'))
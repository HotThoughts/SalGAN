import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from model import *
from util import *

torch.cuda.empty_cache()


def fit(
    train_dl,
    valid_dl,
    G,
    loss_fn,
    g_opt,
    num_epochs,
    curr_epoch,
    device,
    D,
    d_opt,
    batch_size,
):

    for epoch in range(num_epochs):
        # Compute the current epoch
        epoch += curr_epoch + 1
        # Train
        G.train()
        D.train()
        g_train_loss = []
        d_train_loss = []
        for i, batch in enumerate(train_dl):
            img = batch["image"].to(device)
            fix = batch["fixation"].to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Loss_D = L( D(I, S), 1 ) + L( D(I, S^hat), 0 )

            if i % 2 == 1:
                # compute loss
                inputs_real = torch.cat((img, fix), 1)
                inputs_fake = torch.cat((img, G(img)), 1)
                outputs_real = D(inputs_real)
                outputs_fake = D(inputs_fake)

                d_loss_real = loss_fn(outputs_real, real_labels)
                d_loss_fake = loss_fn(outputs_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                # d_train_loss += d_loss
                d_train_loss.append(d_loss.data.item())

                # Backprop and optimize
                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()
                # print(f'Batch {i}/{len(train_dl)}   D loss: {d_loss: .6f}')
            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Loss_G = alpha * BCELoss(outputs, real) + L( D(I, S^hap), 1)

            else:
                fake_fix = G(img)
                fake_inputs = torch.cat((img, fake_fix), 1)
                fake_outputs = D(fake_inputs)

                g_lossG = loss_fn(fake_fix, fix)
                g_lossD = loss_fn(fake_outputs, real_labels)

                alpha = 0.05  # use a value such that maximizing the Information Gain
                g_loss = alpha * g_lossG + g_lossD
                # g_train_loss += g_loss
                g_train_loss.append(g_loss.data.item())

                # Backprop and optimize
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
                # print(f'Batch {i}/{len(train_dl)}   G loss: {g_loss: .6f}')
        # Average training loss
        g_train_loss = sum(g_train_loss) / len(g_train_loss)
        d_train_loss = sum(d_train_loss) / len(d_train_loss)
        # Evaluate
        G.eval()
        with torch.no_grad():
            g_valid_loss = sum(
                loss_fn(G(batch["image"].to(device)), batch["fixation"].to(device))
                for batch in valid_dl
            )
        g_valid_loss /= len(valid_dl)

        # Log info
        print(
            f"Epoch {epoch: 2d},\n   G_train_loss = {g_train_loss: .4f}, G_valid_loss = {g_valid_loss: .4f}\n   D_train_loss = {d_train_loss: .4f}"
        )

        # Save everything every 10 epochs
        PATH = f"/informatik1/students/home/9wei/Documents/SalGAN_checkpoint/exp02/epoch{epoch}.ckpt"
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "G_state_dict": G.state_dict(),
                    "D_state_dict": D.state_dict(),
                    "G_optimizer_state_dict": g_opt.state_dict(),
                    "D_optimizer_state_dict": d_opt.state_dict(),
                    "G_train_loss": g_train_loss,
                    "G_valid_loss": g_valid_loss,
                    "D_train_loss": d_train_loss,
                },
                PATH,
            )
        else:  # Save loss every epoch
            torch.save(
                {
                    "epoch": epoch,
                    "G_train_loss": g_train_loss,
                    "G_valid_loss": g_valid_loss,
                    "D_train_loss": d_train_loss,
                },
                PATH,
            )


def main():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Hyper parameter
    BATCH_SIZE = 8
    ROOT_DIR = "/export/scratch/CV2/"
    TRAIN_DL, VALID_DL = get_data(ROOT_DIR, BATCH_SIZE)
    G = Generator().to(device)
    D = Discriminator().to(device)
    g_opt = torch.optim.Adam(G.parameters(), lr=3e-4, weight_decay=1e-4)
    d_opt = torch.optim.Adam(D.parameters(), lr=3e-4, weight_decay=1e-4)
    LOSS_FN = nn.BCELoss()
    NUM_EPOCHS = 120
    CURR_EPOCH = 0
    RESUME_TRAINING = True

    if RESUME_TRAINING:
        CK_PATH = "/informatik1/students/home/9wei/Documents/SalGAN_checkpoint/exp02/epoch10.ckpt"
        checkpoint = torch.load(CK_PATH)
        G.load_state_dict(checkpoint["model_state_dict"])
        g_opt.load_state_dict(checkpoint["optimizer_state_dict"])
        CURR_EPOCH = checkpoint["epoch"]

    # Start training
    fit(
        TRAIN_DL,
        VALID_DL,
        G,
        LOSS_FN,
        g_opt,
        NUM_EPOCHS,
        CURR_EPOCH,
        device,
        D,
        d_opt,
        BATCH_SIZE,
    )


if __name__ == "__main__":
    main()

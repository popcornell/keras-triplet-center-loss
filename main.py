from train_routines import xentropy, center_loss, center_triplet_loss, triplet_loss
import os

outdir = os.getcwd() + "/runs/"

batch_size = 64
n_epochs = 20
lr = 0.05
loss_weights = [0.8 , 0.2]

xentropy.train(outdir, batch_size, n_epochs, lr)
center_loss.train(outdir, batch_size, n_epochs, lr, loss_weights)
triplet_loss.train(outdir, batch_size, n_epochs, lr, loss_weights)
center_triplet_loss.train(outdir, batch_size, n_epochs, lr, loss_weights)

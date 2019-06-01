from train_routines import xentropy, center_loss, triplet_center_loss, triplet_loss
import os

outdir = os.getcwd() + "/runs/"

batch_size = 64
n_epochs = 10
lr = 0.05
loss_weights = [1, 0.01] #xentropy vs center/triplet loss weights

xentropy.train(outdir, batch_size, n_epochs, lr)
center_loss.train(outdir, batch_size, n_epochs, lr, loss_weights)
triplet_loss.train(outdir, batch_size, n_epochs, lr, loss_weights)
triplet_center_loss.train(outdir, batch_size, n_epochs, lr, loss_weights)



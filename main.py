from train_routines import xentropy, center_loss, triplet_center_loss, triplet_loss
import os

outdir = os.getcwd() + "/runs/"

batch_size = 64
n_epochs = 10
lr = 0.05
loss_weights = [1, 0.01] #xentropy vs center/triplet loss weights
embedding_size = 8


xentropy.train(outdir, batch_size, n_epochs, lr, embedding_size)
# Keras center loss code is from shamangary: https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization
center_loss.train(outdir, batch_size, n_epochs, lr, embedding_size, loss_weights)
# Keras triplet loss code is from KinWaiCheuk: https://github.com/KinWaiCheuk/Triplet-net-keras
triplet_loss.train(outdir, batch_size, n_epochs, lr, embedding_size, loss_weights)
triplet_center_loss.train(outdir, batch_size, n_epochs, lr, embedding_size,  loss_weights)




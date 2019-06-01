##########################################################################
### Original implementation by shamangary: https://github.com/shamangary/Keras-MNIST-center-loss-with-visualization
##########################################################################

from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.models import Model
from load_mnist import load_mnist
from base_network import cnn
from triplet import generate_triplet, triplet_center_loss
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import concatenate, Lambda, Embedding
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import os


def train(outdir, batch_size, n_epochs, lr, embedding_size, loss_weights):
    print("#" * 100)
    print("Training with Center Loss....")
    print("#" * 100)

    outdir = outdir + "/center_loss/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot = load_mnist()

    x_input = Input(shape=(28, 28, 1))

    softmax, pre_logits = cnn(x_input, embedding_size)

    target_input = Input((1,), name='target_input')

    center = Embedding(10, embedding_size)(target_input)
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
        [pre_logits, center])

    model = tf.keras.models.Model(inputs=[x_input, target_input], outputs=[softmax, l2_loss])

    model.compile(loss=["categorical_crossentropy", lambda y_true, y_pred: y_pred],
                  optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=["accuracy"],
                  loss_weights=loss_weights)

    model.fit([x_train, y_train], y=[y_train_onehot, y_train],
              batch_size=batch_size, epochs=n_epochs, callbacks=[TensorBoard(log_dir=outdir)], validation_split=0.2)

    model.save(outdir + "center_loss_model.h5")

    model = Model(inputs=[x_input, target_input], outputs=[softmax, l2_loss, pre_logits])
    model.load_weights(outdir + "center_loss_model.h5")

    _, _, X_train_embed = model.predict([x_train[:512], y_train[:512]])
    _, _, X_test_embed = model.predict([x_test[:512], y_test[:512]])

    from TSNE_plot import tsne_plot

    tsne_plot(outdir, "center_loss", X_train_embed, X_test_embed, y_train, y_test)

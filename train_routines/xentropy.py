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


def train(outdir, batch_size, n_epochs, lr):
    print("Training with Categorical CrossEntropy Only Loss....")

    outdir = outdir + "/xentropy_only_loss/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot = load_mnist()

    x_input = Input(shape=(28, 28, 1))

    softmax, pre_logits = cnn(x_input)


    model = tf.keras.models.Model(inputs=[x_input], outputs=[softmax])

    model.compile(loss=["categorical_crossentropy"],
                  optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=["accuracy"],
                  )

    model.fit([x_train], y=[y_train_onehot],
              batch_size=batch_size, epochs=n_epochs, callbacks=[TensorBoard(log_dir=outdir)], validation_split=0.2)

    model.save(outdir + "xentropy_loss_model.h5")

    model = Model(inputs=[x_input], outputs=[softmax,  pre_logits])
    model.load_weights(outdir + "xentropy_loss_model.h5")

    _,  X_train_embed = model.predict([x_train[:512]])
    _,  X_test_embed = model.predict([x_test[:512]])

    from TSNE_plot import tsne_plot

    tsne_plot(outdir, "xentropy_loss", X_train_embed, X_test_embed, y_train, y_test)

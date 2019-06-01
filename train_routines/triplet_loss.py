from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.models import Model
from load_mnist import load_mnist
from base_network import cnn
from triplet import generate_triplet, triplet_loss
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import concatenate, Lambda, Embedding
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import os


def train(outdir, batch_size, n_epochs, lr, loss_weights):

    print("Training with Triplet Center Loss....")

    outdir = outdir + "/triplet_loss/"

    if not os.path.isdir(outdir):
        os.makedirs(outdir)


    x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot = load_mnist()

    model_input = Input(shape=(28, 28, 1))

    softmax, pre_logits = cnn(model_input)

    x_train_flat = x_train.reshape(-1, 784)

    X_train, Y_train = generate_triplet(x_train_flat, y_train, ap_pairs=150, an_pairs=150)

    target_input = Input((1,), name='target_input')

    shared_model = tf.keras.models.Model(inputs=[model_input, target_input], outputs=[softmax, pre_logits])

    anchor_input = Input((28, 28, 1,), name='anchor_input')
    positive_input = Input((28, 28, 1,), name='positive_input')
    negative_input = Input((28, 28, 1,), name='negative_input')


    soft_anchor, pre_logits_anchor = shared_model([anchor_input])
    soft_pos, pre_logits_pos = shared_model([positive_input])
    soft_neg, pre_logits_neg = shared_model([negative_input])

    merged_pre = concatenate([pre_logits_anchor, pre_logits_pos, pre_logits_neg], axis=-1, name='merged_pre')

    merged_soft = concatenate([soft_anchor, soft_pos, soft_neg], axis=-1, name='merged_soft')

    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=[merged_soft, merged_pre])
    model.compile(loss=["categorical_crossentropy", triplet_loss],
                  optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=["accuracy"], loss_weights = loss_weights
                  )


    le = LabelBinarizer()

    anchor = X_train[:, 0, :].reshape(-1, 28, 28, 1)
    positive = X_train[:, 1, :].reshape(-1, 28, 28, 1)
    negative = X_train[:, 2, :].reshape(-1, 28, 28, 1)

    y_anchor = le.fit_transform(Y_train[:, 0])
    y_positive = le.fit_transform(Y_train[:, 1])
    y_negative = le.fit_transform(Y_train[:, 2])

    target = np.concatenate((y_anchor, y_positive,y_negative), -1)

    model.fit([anchor, positive, negative], y=[target, target],
              batch_size=batch_size, epochs=n_epochs, callbacks=[TensorBoard(log_dir= outdir)], validation_split = 0.2)

    model.save(outdir + "triplet_loss_model.h5")

    model = Model(inputs=[anchor_input], outputs=[soft_anchor, pre_logits_anchor])
    model.load_weights(outdir + "triplet_loss_model.h5")

    _,  X_train_embed = model.predict([x_train[:512]])
    _,  X_test_embed = model.predict([x_test[:512]])

    from TSNE_plot import tsne_plot

    tsne_plot(outdir, "triplet_loss", X_train_embed, X_test_embed,y_train, y_test)


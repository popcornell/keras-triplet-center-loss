# HYPERPARAMS
batch_size = 64
n_epochs = 10


# loading MNIST

from tensorflow.keras.datasets import fashion_mnist

def load_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')


    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist()

from sklearn.preprocessing import LabelBinarizer
le = LabelBinarizer()

y_train_onehot = le.fit_transform(y_train)
y_test_onehot = le.transform(y_test)

x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

from tensorflow.keras.layers import Input
import tensorflow as tf
from tensorflow.keras.models import Model

model_input = Input(shape=(28,28, 1))

from cnn import cnn, generate_triplet, center_triplet_loss

softmax, pre_logits = cnn(model_input)

x_train_flat = x_train.reshape(-1,784)
x_test_flat = x_test.reshape(-1,784)

X_train, Y_train = generate_triplet(x_train_flat,y_train, ap_pairs=150, an_pairs=150)


from tensorflow.keras.layers import concatenate, Lambda, Embedding
import tensorflow.keras.backend as K


target_input = Input((1, ), name='target_input')

center = Embedding(10, 32 )(target_input)
l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss_anchor')([pre_logits, center])


shared_model = tf.keras.models.Model(inputs=[model_input, target_input], outputs=[softmax,  pre_logits])



anchor_input = Input((28,28,1, ), name='anchor_input')
positive_input = Input((28,28,1, ), name='positive_input')
negative_input = Input((28,28,1, ), name='negative_input')


target_anchor_input = Input((1, ), name="anchor_target_input")
target_positive_input = Input((1, ), name='target_positive_input')
target_negative_input = Input((1, ), name='target_negative_input')


soft_anchor,  pre_logits_anchor = shared_model([anchor_input, target_anchor_input])
soft_pos, pre_logits_pos = shared_model([positive_input, target_positive_input])
soft_neg, pre_logits_neg = shared_model([negative_input, target_negative_input])

# center triplet loss

#center_anchor = Embedding(10, 32 )(target_anchor_input)
#center_pos = Embedding(10, 32 )(target_positive_input)
#center_neg = Embedding(10, 32 )(target_negative_input)

#l2_loss_anchor = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss_anchor')([encoded_anchor, center_anchor])
#l2_loss_pos = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss_pos')([encoded_positive, center_pos])
#l2_loss_neg = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss_neg')([encoded_negative, center_neg])

#merged_l2 = concatenate([l2_anchor, l2_positive, l2_negative], axis=-1, name="merged_l2")
#merged_centers = concatenate([ center_anchor, center_pos, center_neg], axis=-1, name='merged_centers')
#merged_encodings = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_encodings')

merged_soft = concatenate([soft_anchor, soft_pos, soft_neg], axis=-1, name='merged_soft')


model = Model(inputs=[anchor_input, positive_input, negative_input, target_anchor_input, target_positive_input, target_negative_input], outputs=[merged_soft])
model.compile(loss=["categorical_crossentropy"], optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=["accuracy"],
              )

import numpy as np

Anchor = X_train[:,0,:].reshape(-1,28,28,1)
Positive = X_train[:,1,:].reshape(-1,28,28,1)
Negative = X_train[:,2,:].reshape(-1,28,28,1)

Y_Anchor = le.fit_transform(Y_train[:, 0])
Y_Positive = le.fit_transform(Y_train[:, 1])
Y_Negative = le.fit_transform(Y_train[:, 2])


target = np.concatenate((Y_Anchor, Y_Positive, Y_Negative), -1)


model.fit([Anchor, Positive, Negative, Y_train[:, 0], Y_train[:, 1], Y_train[:, 2] ],y=[target], batch_size=batch_size, epochs=n_epochs)

model.save("tripletloss_xentropy_model.h5")

model = Model(inputs=[anchor_input, target_anchor_input], outputs= [soft_anchor, l2_anchor, pre_logits_anchor])
model.load_weights("tripletloss_xentropy_model.h5")

_, _, X_train_embed = model.predict([x_train[:512], y_train[:512]])
_, _, X_test_embed = model.predict([x_test[:512], y_train[:512]])

from TSNE_plot import tsne_plot

tsne_plot(X_train_embed, y_train, X_test_embed, y_test, "xentropy triplet")




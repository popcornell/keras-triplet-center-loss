# HYPERPARAMS
batch_size = 64
n_epochs = 30


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

from cnn import cnn, generate_triplet, triplet_loss

softmax, pre_logits = cnn(model_input)

x_train_flat = x_train.reshape(-1,784)
x_test_flat = x_test.reshape(-1,784)

X_train, Y_train = generate_triplet(x_train_flat,y_train, ap_pairs=150, an_pairs=150)

shared_model = tf.keras.models.Model(inputs=model_input, outputs=[softmax, pre_logits])

from tensorflow.keras.layers import concatenate

anchor_input = Input((28,28,1, ), name='anchor_input')
positive_input = Input((28,28,1, ), name='positive_input')
negative_input = Input((28,28,1, ), name='negative_input')

soft_anchor, encoded_anchor = shared_model(anchor_input)
soft_pos, encoded_positive = shared_model(positive_input)
soft_neg, encoded_negative = shared_model(negative_input)

merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
merged_soft = concatenate([soft_anchor, soft_pos, soft_neg], axis=-1, name='merged_soft')

def xentropy(y_true, y_pred):

    total_lenght = y_pred.shape.as_list()[-1]

    anchor_pred = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive_pred = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative_pred = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    anchor_true = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive_true = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative_true = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]


model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=[merged_vector, merged_soft])
model.compile(loss=[triplet_loss, "categorical_crossentropy"], optimizer=tf.keras.optimizers.Adam(lr=0.05), metrics=["accuracy"],
              loss_weights=[0.7, 0.3])


import numpy as np

Anchor = X_train[:,0,:].reshape(-1,28,28,1)
Positive = X_train[:,1,:].reshape(-1,28,28,1)
Negative = X_train[:,2,:].reshape(-1,28,28,1)

Y_Anchor = le.fit_transform(Y_train[:, 0])
Y_Positive = le.fit_transform(Y_train[:, 1])
Y_Negative = le.fit_transform(Y_train[:, 2])


target = np.concatenate((Y_Anchor, Y_Positive, Y_Negative), -1)


model.fit([Anchor, Positive, Negative],y=[target, target], batch_size=batch_size, epochs=n_epochs)





# plain base network
from tensorflow.keras.layers import Input,Dense, Flatten,  BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.models import Model
import tensorflow as tf

def cnn(input
       ):


    net = Conv2D(16, kernel_size=(3, 3),
                     )(input)

    net = BatchNormalization()(net)

    net = ReLU()(net)

    net = Conv2D(32, (3, 3), strides=(2,2))(net)

    net = BatchNormalization()(net)

    net = ReLU()(net)

    net = Flatten()(net)
    net = Dense(32)(net)
    net = BatchNormalization()(net)
    pre_logit = ReLU()(net)
    softmax = Dense(10, activation='softmax')(pre_logit)

    return softmax, pre_logit


# triplet loss
from itertools import permutations
import random

import numpy as np

def generate_triplet(x, y,  ap_pairs=10, an_pairs=10):
    data_xy = tuple([x, y])

    trainsize = 1

    triplet_train_pairs = []
    y_triplet_pairs = []
    #triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        A_P_pairs = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)  # Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        #Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            y_Anchor = data_xy[1][ap[0]]
            Positive = data_xy[0][ap[1]]
            y_Pos = data_xy[1][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                y_Neg = data_xy[1][n]
                triplet_train_pairs.append([Anchor, Positive, Negative])
                y_triplet_pairs.append([y_Anchor, y_Pos, y_Neg])
                # test

    return np.array(triplet_train_pairs), np.array(y_triplet_pairs)

import tensorflow.keras.backend as K

def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss
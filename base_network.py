# plain base network
from tensorflow.keras.layers import Input,Dense, Flatten,  BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.models import Model
import tensorflow as tf

def cnn(input
       ):

    net = Conv2D(2, kernel_size=(3, 3),
                     )(input)

    net = BatchNormalization()(net)

    net = ReLU()(net)

    net = Conv2D(4, (3, 3), strides=(2,2))(net)

    net = BatchNormalization()(net)

    net = ReLU()(net)

    net = Flatten()(net)
    net = Dense(8)(net)
    net = BatchNormalization()(net)
    pre_logit = ReLU()(net)
    softmax = Dense(10, activation='softmax')(pre_logit)

    return softmax, pre_logit



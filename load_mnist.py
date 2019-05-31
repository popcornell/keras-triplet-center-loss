from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import fashion_mnist as mnist


def load_mnist():

    def _load_mnist(fashion = False):

        if fashion:
            from tensorflow.keras.datasets import fashion_mnist as mnist

        else:

            from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        return x_train, y_train, x_test, y_test

    x_train, y_train, x_test, y_test = _load_mnist()


    le = LabelBinarizer()

    y_train_onehot = le.fit_transform(y_train)
    y_test_onehot = le.transform(y_test)

    x_train = x_train.reshape(*x_train.shape, 1)
    x_test = x_test.reshape(*x_test.shape, 1)

    return x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot


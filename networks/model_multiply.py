import h5py
import keras
import random
from function.data_generator import DataGenerator
from layers.attention import Attention
from layers.transformer import Transformer
from keras.layers import Input, Dense, Permute, Reshape
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Add, Average, Multiply
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
import matplotlib.pylab as pl

'''
本代码将Maximum换成Average
将Maxout变为对应项相加再平均
'''

epochs = 100
batch_size = 64
frames = 30
learning_rate = 0.001
d_k = 30
input_shape = (frames, 25, 3)
use_bias = True
train_path = 'F:/NTU60/data_set/cross_subject/cs_trn.hdf5'
tst_path = 'F:/NTU60/data_set/cross_subject/cs_tst.hdf5'
weight_path = 'F:/NTU60/weights/second/multiply.h5'
graph_path = 'F:/NTU60/model_v1.png'


# 跑通的代码
# 共享层的写法Model是可以分部分来写的
def share_stream(x_shape):
    x = Input(shape=x_shape)
    x_a = Transformer(d_k, frames)(x)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   use_bias=use_bias)(x_a)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   use_bias=use_bias)(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   use_bias=use_bias)(conv2)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3)

    shared_layer = Model(x, conv3)
    return shared_layer


def model():
    up_0 = Input(shape=input_shape, name='up_stream_0')
    up_1 = Input(shape=input_shape, name='up_stream_1')
    down_0 = Input(shape=input_shape, name='down_stream_0')
    down_1 = Input(shape=input_shape, name='down_stream_1')

    up_stream = share_stream(x_shape=input_shape)
    down_stream = share_stream(x_shape=input_shape)

    up_feature_0 = up_stream(up_0)
    up_feature_1 = up_stream(up_1)
    down_feature_0 = down_stream(down_0)
    down_feature_1 = down_stream(down_1)

    up_feature_0 = Flatten()(up_feature_0)
    up_feature_1 = Flatten()(up_feature_1)
    down_feature_0 = Flatten()(down_feature_0)
    down_feature_1 = Flatten()(down_feature_1)

    up_feature = Multiply()([up_feature_0, up_feature_1])
    down_feature = Multiply()([down_feature_0, down_feature_1])

    feature = concatenate([up_feature, down_feature])

    fc_1 = Dense(units=256, activation='relu', kernel_regularizer=l2(0.001))(feature)
    fc_1 = Dropout(0.5)(fc_1)

    fc_2 = Dense(units=60, activation='softmax')(fc_1)

    network = Model(input=[up_0, up_1, down_0, down_1], outputs=fc_2)
    return network


def train_model(network):
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    network.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    network.summary()
    plot_model(network, to_file=graph_path)

    batch_num = 0
    model_save_acc = 0
    all_train_accuracy = []
    all_train_loss = []
    all_tst_accuracy = []

    tst_data = DataGenerator(h5file_path=tst_path, batch_size=batch_size, frames=frames)
    tst_data_name = tst_data.get_data_name()
    tst_cursors = [tst_data.get_cursors(name) for name in tst_data_name]

    for epoch in range(epochs):
        accuracy_list = []
        loss_list = []
        print(epoch + 1, ' epoch is beginning......')
        train_data = DataGenerator(h5file_path=train_path, batch_size=batch_size, frames=frames)
        train_data_name = train_data.get_data_name()
        train_data_cursors = train_data.batch_cursors(len(train_data_name))
        index_num = random.sample(range(len(train_data_cursors)), len(train_data_cursors))

        for ind in index_num:
            batch_num += 1
            up_data_0, down_data_0, train_labels_0 \
                = train_data.generate_batch_data(train_data_name, train_data_cursors[ind], 0)
            up_data_1, down_data_1, train_labels_1 \
                = train_data.generate_batch_data(train_data_name, train_data_cursors[ind], 1)
            train_loss = network.train_on_batch([up_data_0, up_data_1, down_data_0, down_data_1], train_labels_0)
            accuracy_list.append(train_loss[1])
            loss_list.append(train_loss[0])
            if batch_num % 50 == 0:
                print('the %r batch: loss: %r  accuracy: %r' % (batch_num, train_loss[0], train_loss[1]))

        epoch_accuracy = sum(accuracy_list) / len(accuracy_list)
        epoch_loss = sum(loss_list) / len(loss_list)
        all_train_accuracy.append(epoch_accuracy)
        all_train_loss.append(epoch_loss)

        print('the %r epoch: mean loss: %r    mean accuracy: %r' % (epoch + 1, epoch_loss, epoch_accuracy))

        if epoch >= 0:
            tst_accuracy_list = []
            for num in range(len(tst_data_name)):
                tst_up_0, tst_down_0, tst_labels_0 = \
                    tst_data.get_tst_single_data(tst_data_name[num], tst_cursors[num], 0)
                tst_up_1, tst_down_1, tst_labels_1 = \
                    tst_data.get_tst_single_data(tst_data_name[num], tst_cursors[num], 1)
                tst_loss = network.test_on_batch([tst_up_0, tst_up_1, tst_down_0, tst_down_1], tst_labels_0)
                tst_accuracy_list.append(tst_loss[1])
            tst_accuracy = sum(tst_accuracy_list) / len(tst_accuracy_list)
            all_tst_accuracy.append(tst_accuracy)
            print('The test data accuracy: %r' % tst_accuracy)
            if tst_accuracy > model_save_acc:
                network.save_weights(weight_path)
                model_save_acc = tst_accuracy

    pl.figure()
    trn_acc = pl.subplot(2, 2, 1)
    trn_loss = pl.subplot(2, 2, 2)
    tst_acc = pl.subplot(2, 1, 2)

    pl.sca(trn_acc)
    pl.plot(range(len(all_train_accuracy)), all_train_accuracy, label='train accuracy')
    pl.xlabel('Epoch')
    pl.ylabel('Accuracy')
    pl.ylim(0, 1.0)

    pl.sca(trn_loss)
    pl.plot(range(len(all_train_loss)), all_train_loss, label='loss')
    pl.xlabel('Epoch')
    pl.ylabel('Loss')
    pl.ylim(0, 5.0)

    pl.sca(tst_acc)
    pl.plot(range(len(all_tst_accuracy)), all_tst_accuracy, label='test accuracy')
    pl.xlabel('Epoch')
    pl.ylabel('Accuracy')
    pl.ylim(0, 1.0)

    pl.legend()
    pl.show()


if __name__ == '__main__':
    network = model()
    train_model(network)


from keras.layers import Input, Conv2D, MaxPooling2D, MaxPool3D, Dense,Flatten,Lambda
from keras import backend as k
from keras.engine.topology import Layer
from keras.models import Model
import tensorflow as tf
import numpy as np
from preprocessing import BatchGenerator
from utils import to_index_list
from keras.optimizers import SGD


class FPNN():
    def __init__(self):
        input1 = Input(shape=(64,32,3))
        x1 = Conv2D(64,5,strides=(1,1),padding='valid',activation='relu', use_bias=True)(input1)
        x1 = MaxPooling2D(pool_size=(3,3), strides=3, padding='valid')(x1)
        input2 = Input(shape=(64,32,3))
        x2 = Conv2D(64,5,strides=(1,1),padding='valid',activation='relu', use_bias=True)(input2)
        x2 = MaxPooling2D(pool_size=(3,3), strides=3, padding='valid')(x2)
        x3 = patch_match_layer()([x1,x2])
        x4 = Lambda(MaxOut)(x3)
        x5 = Conv_New(16,3,20,(1,1))(x4)
        x6 = Flatten()(x5)
        x7 = Dense(128,)(x6)
        x8 = Dense(2,activation='softmax')(x7)
        self.model = Model(inputs=[input1, input2], outputs=x8)
        #model.summary()

    def train (self, id_list_dir, val_id_list_dir, train_or_val, batch_size, shuffle, epoch_size):

        id_data = np.load(id_list_dir)
        val_id_list = np.load(val_id_list_dir)
        id_list = to_index_list(id_data, val_id_list, 'train', 3) # the constant here represents the desired set for validation
        training_generator = BatchGenerator(id_data,
                               id_list,
                               val_id_list,
                               'train',
                               batch_size,
                               shuffle,
                               epoch_size,
                               3)
        id_list = to_index_list(id_data, val_id_list, 'val',
                                3)  # the constant here represents the desired set for validation
        validation_generator = BatchGenerator(id_data,
                                       id_list,
                                       val_id_list,
                                       'val',
                                       batch_size,
                                       shuffle,
                                       epoch_size,
                                       3)
        optimizer = SGD(lr=0.01, momentum=0.005, decay=0.001)
        self.model.compile(loss= 'binary_crossentropy', optimizer=optimizer)
        self.model.fit_generator(generator=training_generator,steps_per_epoch=300,epochs=10,validation_data=validation_generator, validation_steps= len(validation_generator)/512) # DO NOT FORGET TO USE VALIDATION SETP










class patch_match_layer(Layer):
    def __init__(self , **kwargs):

        super(patch_match_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        super(patch_match_layer, self).build(input_shape)

    def call(self, inputs):
        x1 = tf.transpose(inputs[0], (0, 3, 2, 1 ))
        x2 = tf.transpose(inputs[1], (0, 3, 1, 2 )) # on 29/10/2018 I changed the index in inputs[index] from 0 to 1 , and I don't know why
        #x3 = tf.expand_dims(x1,1)
        for i in range(int(x1.shape[3].value) ):
            a = x1[:, :, :, i:i+1]
            b = x2[:, :, i:i+1,:]
            x3 = tf.matmul(a,b)
            if i == 0 :
                x4 = tf.expand_dims(x3,2)
            else :
                x4_1 = tf.expand_dims(x3,2)
                x4 = tf.concat([x4,x4_1], axis=2)
        return x4

    def compute_output_shape(self, input_shape):
        print((input_shape[0][0],64,20,9,9))
        return (input_shape[0][0],64,20,9,9)


def MaxOut(x3):
    x4 = MaxPool3D(pool_size=(4, 1, 1), strides=(4, 1, 1), padding='valid')(x3[1:2, ])
    x4 = tf.transpose(x4,[0,2,1,3,4])

    return x4

class Conv_New (Layer) :
    def __init__(self, output_channels, filter_size, stripes_dim, stride, **kwargs):
        self.output_channels = output_channels
        self.filter_size = filter_size
        self.stride = stride
        self.stripes_dim = stripes_dim
        super(Conv_New, self).__init__(**kwargs)

    def build(self, input_shape):

        kernel_shape = (self.stripes_dim,) + (self.output_channels,) + (16,) + (self.filter_size, self.filter_size)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      name='kernel',
                                      initializer='glorot_uniform',
                                      )
        super(Conv_New, self).build(input_shape)

    def call(self, inputs):

        for i in range(self.stripes_dim):
            b = inputs[:, i: i+1, :, :, :]
            c = self.kernel[i: i+1, :, :, :]
            b = k.squeeze(b, 1)
            c = k.squeeze(c, 0)
            b = tf.transpose(b, [0, 2, 3, 1])
            c = tf.transpose(c, [2, 3, 0, 1])
            outputs_1 = k.conv2d(
                b,
                c,
                strides=self.stride,
                padding='valid'
                )
            if i == 0 :
                outputs = tf.expand_dims(outputs_1, 1)
            else :
                outputs_1 = tf.expand_dims(outputs_1, 1)
                outputs = tf.concat([outputs,outputs_1], axis=1)
        return outputs

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[1], input_shape[2], 7, 7)

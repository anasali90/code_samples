from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, ZeroPadding2D, Dropout, Lambda, Activation, \
    Conv2DTranspose, Concatenate, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import SGD, Adam
import tensorflow as tf
from keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
import pickle as pkl
from keras.initializers import Initializer
from keras import initializers
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model
a = np.load('pre_trained_model.npy')


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''

class BInitilizer(Initializer):
    """Initializer that generates tensors initialized to by pre-trained model.
    """

    def __init__(self, name=None, slice=None,weights=None):

        self.name = name
        self.slice = slice
        self.weights = 0
        for i in pre_trained_weights:
            if i['name'] == name:
                if self.slice == None:
                    self.weights = i['weights'][1]
                elif self.slice == 0:
                    shape = int(i['weights'][1].shape[0] / 2)
                    self.weights = i['weights'][1][0:shape]
                else:
                    shape = int(i['weights'][1].shape[0] / 2)
                    self.weights = i['weights'][1][shape:]

    def __call__(self, shape, dtype=None):
        if type(shape) == tuple:
                return K.constant(self.weights.reshape(shape))
        else:
                return K.constant(self.weights.reshape((shape.shape[0].value,)))

    def get_config(self):
        return {
            'name' :self.name,
            'slice':self.slice,
            'weights': self.weights
        }


class KInitilizer(Initializer):
    """Initializer that generates tensors initialized to by pre-trained model.
    """

    def __init__(self, name=None, slice=None, weights=None):

        self.name = name
        self.slice = slice
        self.weights = 0
        for i in pre_trained_weights:
            if i['name'] == name:
                if self.slice == None:
                   self.weights = i['weights'][0]
                elif self.slice == 0:
                    shape = int(i['weights'][0].shape[0]/2)
                    self.weights = i['weights'][0][0:shape,:,:,:]
                else:
                    shape = int(i['weights'][0].shape[0] / 2)
                    self.weights = i['weights'][0][shape:, :, :, :]


    def __call__(self, shape, dtype=None):
        if type(shape) == tuple:
            return K.constant(self.weights.reshape(shape))
        else:
            return K.constant(self.weights.reshape((shape.shape[0].value,shape.shape[1].value,shape.shape[2].value,shape.shape[3].value)))

    def get_config(self):
        return {
             'name' :self.name,
             'slice':self.slice,
             'weights': self.weights
        }


'''

class AlexNetLayout():
    def __init__(self):
        input1 = Input(shape=(240, 320, 3))
        conv1 = Conv2D(96, 11, strides=2, activation='relu', use_bias=True, weights=a[0,:])(input1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=3, strides=2)(conv1)


        strip_1 = Lambda(lambda x: x[:, :, :, :48])(pool1)
        strip_2 = Lambda(lambda x: x[:, :, :, 48:])(pool1)
        conv2_1 = Conv2D(128, 5, strides=1, padding='same', activation='relu', use_bias=True,weights=a[1,:])

        conv2_2 = Conv2D(128, 5, strides=1, padding='same', activation='relu', use_bias=True,weights=a[2,:])

        conv_strip_1 = conv2_1(strip_1)

        conv_strip_2 = conv2_2(strip_2)

        conv2_output = Concatenate(axis=-1)([conv_strip_1, conv_strip_2])
        conv2_output = BatchNormalization()(conv2_output)
        pool2 = MaxPooling2D(pool_size=3, strides=2)(conv2_output)
        ################################################################################################################
        conv3 = Conv2D(384, 3, padding='same', activation='relu', use_bias=True,weights=a[3,:])(pool2)
        conv3 = BatchNormalization()(conv3)
        ################################################################################################################
        strip_1 = Lambda(lambda x: x[:, :, :, :192])(conv3)

        strip_2 = Lambda(lambda x: x[:, :, :, 192:])(conv3)
        conv4_1 = Conv2D(192, 3, padding='same', activation='relu', use_bias=True,weights=a[4,:])

        conv4_2 = Conv2D(192, 3, padding='same', activation='relu', use_bias=True,weights=a[5,:])
        conv_strip_1 = conv4_1(strip_1)

        conv_strip_2 = conv4_2(strip_2)

        conv4_output = Concatenate(axis=-1)([conv_strip_1, conv_strip_2])
        conv4_output = BatchNormalization()(conv4_output)
        ################################################################################################################
        strip_1 = Lambda(lambda x: x[:, :, :, :192])(conv4_output)
        strip_2 = Lambda(lambda x: x[:, :, :, 192:])(conv4_output)
        conv5_1 = Conv2D(128, 3, padding='same', activation='relu', use_bias=True,weights=a[6,:])
        conv5_2 = Conv2D(128, 3, padding='same', activation='relu', use_bias=True,weights=a[7,:])
        conv_strip_1 = conv5_1(strip_1)

        conv_strip_2 = conv5_2(strip_2)

        conv5_output = Concatenate(axis=-1)([conv_strip_1, conv_strip_2])
        conv5_output = BatchNormalization()(conv5_output)
        ################################################################################################################
        pool5 = MaxPooling2D(pool_size=2, strides=2)(conv5_output)
        conv6 = Conv2D(4096, 6, padding='same', activation='relu', use_bias=True)(pool5)
        conv6 = BatchNormalization()(conv6)
        drop_c6 = Dropout(rate=0.5)(conv6)
        conv7 = Conv2D(4096, 1, padding='same', activation='relu', use_bias=True)(drop_c6)
        conv7 = BatchNormalization()(conv7)
        drop_c7 = Dropout(rate=0.5)(conv7)
        ################################################################################################################
        conv8 = Conv2D(2, 1, padding='same', activation='relu', use_bias=True)(drop_c7)
        conv8 = BatchNormalization()(conv8)

        '''
        tconv = Conv2DTranspose(2, kernel_size=(2,2),dilation_rate=19,strides=(1,1), activation='relu')(conv8)
        tconv = Lambda(lambda x: x[:, 3:31, :, :])(tconv)
        #tconv = BatchNormalization()(tconv)
        conv4_predctions = Conv2D(2, 1, activation='relu')(conv4_output)
        # conv4_predctions = BatchNormalization()(conv4_predctions)
        scale_1 = Add()([conv4_predctions, tconv])
        #scale_1 = BatchNormalization()(scale_1)
        conv1_predctions = Conv2D(2, 1, activation='relu')(pool1)
        conv1_predctions = ZeroPadding2D(padding=(1, 1))(conv1_predctions)
        conv1_predctions = Lambda(self.splits)(conv1_predctions)
        #conv1_predctions = BatchNormalization()(conv1_predctions)
        tconv1 = Conv2DTranspose(2, kernel_size=(8, 11), dilation_rate=4, activation='relu')(scale_1)
        tconv1 = Lambda(lambda x: x[:, :, 1:77, :])(tconv1)
        tconv1 = ZeroPadding2D(padding=(1, 1))(tconv1)
        #tconv1 = BatchNormalization()(tconv1)
        scale_2 = Add()([conv1_predctions, tconv1])
        output = Conv2DTranspose(2, kernel_size=(17, 22), dilation_rate=12, activation='relu')(scale_2)
        output = Lambda(lambda x: x[:, 5:245, 5:325, :])(output)
        #output = ZeroPadding2D(padding=(4, 4))(output)
        #output = BatchNormalization()(output)
        output1 = Activation('softmax')(output)
        '''
        conv8_upsampled1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8)
        conv4_predctions = Conv2D(2, 1, activation='relu')(conv4_output)
        conv4_predctions = BatchNormalization()(conv4_predctions)
        scale_1 = Add()([conv4_predctions, conv8_upsampled1])


        conv1_predctions = Conv2D(2, 1, activation='relu')(pool1)
        conv1_predctions = BatchNormalization()(conv1_predctions)
        conv1_predctions = ZeroPadding2D(padding=(1, 1))(conv1_predctions)
        conv1_predctions = Lambda(lambda x: x[:,:-1,:-1])(conv1_predctions)
        # scale_1_pad = ZeroPadding2D(padding=(1,0))(scale_1)
        scale_1_upsampled = UpSampling2D(size=(2, 2), interpolation='bilinear')(scale_1)
        scale_1_upsampled = ZeroPadding2D(padding=(1, 1))(scale_1_upsampled)
        scale_2 = Add()([conv1_predctions, scale_1_upsampled])

        scale_2 = ZeroPadding2D(padding=(1, 1))(scale_2)
        #scale_2 = BatchNormalization()(scale_2)
        output = UpSampling2D(size=(4, 4), interpolation='bilinear')(scale_2)
        output1 = Activation('softmax')(output)
        #output1 =BatchNormalization()(output1)
        self.model = Model(inputs=input1, outputs=output1)
        self.model.summary()
    '''
    def splits(self, a):
        return a[:, :-1, :-1]
    def splits_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]-1, input_shape[2]-1, input_shape[3])
   

    def custome_loss(self, y_true, y_pred):
        if y_pred is not None:
           y_pred = tf.reshape(y_pred, [-1])
           y_true = tf.reshape(y_true, [-1])

           return binary_crossentropy(y_true, y_pred)
     '''

    def train(self):
        images_train_dir = 'images_train.npy'
        labels_train_dir = 'labels_train.npy'

        images_train = np.load(images_train_dir)
      #  images_train = images_train/255
        labels_train = np.load(labels_train_dir)

      #  images_test = images_test

        data_generatot = ImageDataGenerator(validation_split=0.2)
        train_generatot = data_generatot.flow(images_train, labels_train, batch_size=16, shuffle=True, subset="training")
        test_generator= data_generatot.flow(images_train, labels_train, batch_size=16, shuffle=True, subset="validation")


        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        #tensorboard = TensorBoard(log_dir='./logs', batch_size=16)

        checkpointer = ModelCheckpoint(filepath='new_weights.hdf5', verbose=1, save_best_only=True)

        self.model.fit_generator(generator=train_generatot, epochs=300, validation_data=test_generator, validation_steps=10, steps_per_epoch=45, callbacks=[checkpointer])
        '''
        for epoch in range(0,300):
            batch = 0
            for i in range(0, int(images_train.shape[0]/16)):

                train = images_train[i:i + 16, :, :, :]
                test = labels_train[i: i + 16, :, :, :]
                loss = self.model.train_on_batch(train, test)
                print('trainig batch num#', i, ' epoch num', epoch, ' Loss is', loss)
        '''



        for i in range(1, 15):
            x = train_generatot.x[i]
            x = x.reshape([1,240,320,3])
         #   x = x/255
           # y = train_generatot.y[i]
          #  y = y.reshape([1, 240, 320, 2])
            #            y = y.reshape([10, 1, 1, 2])
           # v = self.model.test_on_batch(x, y)
            nn = self.model.predict_on_batch(x)
            plt.imshow(nn[0,:,:,0])
            plt.show()

    def evaluate(self, test_images, test_labels):
        print('evaluating: Pixelwise_accuracy')
        predected_gp = self.model.predict_on_batch(test_images)
        predected_gp_label = (predected_gp[:,:,:,1] >0.5)*1
        pixelwise_acc = sum(sum(sum(((predected_gp_label == test_labels[:,:,:,1])*1))))/(test_labels.shape[0]*test_labels.shape[1]*test_labels.shape[2])
        print('acc:', pixelwise_acc)










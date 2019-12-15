import keras
import numpy as np
#from model import KInitilizer, BInitilizer, AlexNetLayout
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
#model11 = load_model('weights_1.hdf5',custom_objects={"BInitilizer": BInitilizer, "KInitilizer":KInitilizer})
model11 = load_model('new_weights_6.hdf5',compile=False)
model11.summary()
images_test_dir = 'images_test.npy'
labels_test_dir = 'labels_test.npy'

test_images = np.load(images_test_dir)
test_labels = np.load(labels_test_dir)


for i in range(0,10):
    print('23434')
    predected_gp = model11.predict_on_batch(test_images[:,:,:,:])
    print('1')
    predected_gp_label = (predected_gp[:, 5:-10, 5:-10, 1] > (i/10)) * 1
    pixelwise_acc = sum(sum(sum(((predected_gp_label == test_labels[:, 5:230, 5:310, 1]) * 1)))) / (
            test_labels.shape[0] *( test_labels.shape[1]-15) * (test_labels.shape[2]-15))
    print(pixelwise_acc)
for i in range(0, 300):
    x = test_images[i,:,:,:]
    x = x.reshape([1, 240, 320, 3])
    #   x = x/255
    # y = train_generatot.y[i]
    #  y = y.reshape([1, 240, 320, 2])
    #            y = y.reshape([10, 1, 1, 2])
    # v = self.model.test_on_batch(x, y)
    nn = model11.predict_on_batch(x)


    plt.subplot(3, 1, 1)
    plt.imshow(nn[0, :, :, 0])


    plt.subplot(3, 1, 2)
    plt.imshow(test_labels[i, :, :, 0])

    plt.subplot(3, 1, 3)
    plt.imshow(x[0,:,:,:])

    plt.show()

    ''' 
    
    plt.imshow(nn[0, :, :, 0])
    plt.show()
    plt.imshow(test_labels[i, :, :, 0])
    plt.show()
    '''

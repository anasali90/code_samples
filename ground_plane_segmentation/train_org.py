import model
import numpy as np

mymodel = model.AlexNetLayout()


images_test_dir = 'images_test.npy'
labels_test_dir = 'labels_test.npy'
mymodel.train()
images_test = np.load(images_test_dir)
labels_test = np.load(labels_test_dir)

mymodel.evaluate(images_test, labels_test)


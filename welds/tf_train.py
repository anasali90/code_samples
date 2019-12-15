import keras 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
image_size = [64,64]
batch_size = 100
epochs = 200 
iterations = 25
data_gen = ImageDataGenerator(
                                horizontal_flip=True,
                                vertical_flip=True,
                                
                                
                                validation_split=0.2
                                )
train_gen = data_gen.flow_from_directory('./data/dataset', class_mode='categorical',batch_size= batch_size, subset="training", target_size=(64, 64))
test_gen = data_gen.flow_from_directory('./data/dataset', class_mode='categorical', subset="validation",batch_size= 1000, target_size=(64, 64)) 
x_test, y_test = next(test_gen)
weights = {
    'w1' : tf.Variable(tf.random_normal([7,7,3, 32])),
    'w2' : tf.Variable(tf.random_normal([7,7,32, 32])),
    'w3' : tf.Variable(tf.random_normal([7,7,32, 32])),
    'w4' : tf.Variable(tf.random_normal([7,7,32, 32])),
    'w5' : tf.Variable(tf.random_normal([4*4*32,2]))   
}
biases = {
    'b1' : tf.Variable(tf.random_normal([32])),
    'b2' : tf.Variable(tf.random_normal([32])),
    'b3' : tf.Variable(tf.random_normal([32])),
    'b4' : tf.Variable(tf.random_normal([32])),
    'b5' : tf.Variable(tf.random_normal([2]))
}

x_train = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], 3])
y_out = tf.placeholder(tf.float32, [None, 2])
#x = tf.image.rgb_to_grayscale(x_train)
#x = tf.Print(x, [x], 'first')
x = tf.nn.conv2d(x_train, filter=weights['w1'], strides=[1,1,1,1], padding='SAME')
x = tf.nn.bias_add(x, biases['b1'])
x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
x = tf.nn.relu(x)


#x = tf.Print(x, [ weights['w1'], weights['w2']], 'weights')
x = tf.nn.conv2d(x, filter=weights['w2'], strides=[1,1,1,1], padding='SAME')
x = tf.nn.bias_add(x, biases['b2'])
x = tf.layers.batch_normalization(x, training=True)
x = tf.nn.relu(x)
x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.nn.conv2d(x, filter=weights['w3'], strides=[1,1,1,1], padding='SAME')
x = tf.nn.bias_add(x, biases['b3'])
x = tf.layers.batch_normalization(x, training=True)
x = tf.nn.relu(x)
x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.nn.conv2d(x, filter=weights['w4'], strides=[1,1,1,1], padding='SAME')
x = tf.nn.bias_add(x, biases['b4'])
x = tf.layers.batch_normalization(x, training=True)
x = tf.nn.relu(x)
x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



x = tf.reshape(x, [-1, 4*4*32])
x = tf.nn.dropout(x, 0.5)
#x = tf.Print(x, [x], 'asdff')
x = tf.matmul(x, weights['w5'])
#x = tf.Print(x, [x, weights['w4'][0,:,:,0]], 'first')
x = tf.nn.bias_add(x, biases['b5'])
x = tf.layers.batch_normalization(x, training=True)
#ff=tf.Print(ff, [ff], 'ff')
y_pred = tf.nn.softmax(x)
correct_pred = tf.equal(tf.argmax(y_pred	, 1), tf.argmax(y_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_out, logits=x)
loss = tf.reduce_mean(cost)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01, ).minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess : 
    sess.run(init)
    
    for eposh in range(epochs) : 
        train_gen = data_gen.flow_from_directory('./data/dataset', class_mode='categorical',batch_size= batch_size, subset="training", target_size=(64, 64))
        print('test results',sess.run([loss, accuracy],  feed_dict={x_train:x_test, y_out:y_test}))
        for i in range(iterations) : 
            x_batch,y_batch = next(train_gen)
            x_batch = x_batch/255
            print('asdf',)
            sess.run(opt, feed_dict={x_train:x_batch, y_out:y_batch})
            #print(sess.run(loss, feed_dict={x_train:x_batch/255, y_out:y_batch}))
            los, acc = sess.run([loss, accuracy],  feed_dict={x_train:x_batch, y_out:y_batch})
            #print(sess.run(opt, feed_dict={x_train:x_batch, labels:y_batch}))
            print("batch " + str(1) + ", Minibatch Loss= " + \
                  "{:.4f}".format(los) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    saver.save(sess, './trained_model')


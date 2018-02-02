import pickle
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

#training parameters
nb_epochs = 10
rate = 0.001
batch_size = 128
debug_test_size = -1         

#classes info
nb_classes = 43
sign_names = pd.read_csv('signnames.csv')

# TODO: Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)
    
# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(data['features'], 
            data['labels'], test_size=0.33)


print('Total size for train %i and test %i, using %i in this run' % (len(y_train), len(y_test), debug_test_size))
if debug_test_size > 0:
    X_train = X_train[:debug_test_size]
    y_train = y_train[:debug_test_size]
    X_test = X_test[:debug_test_size]
    y_test = y_test[:debug_test_size]
            
# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
labels = tf.placeholder(tf.int64, shape=(None))
resized = tf.image.resize_images(features, size=(227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
fc8_shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8_W = tf.Variable(tf.truncated_normal(fc8_shape), name='fc8_weights')
fc8_b = tf.Variable(tf.zeros(nb_classes), name='fc8_biases')
fc8 = tf.add(tf.matmul(fc7, fc8_W), fc8_b)
softmax = tf.nn.softmax(fc8,name='softmax')

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc8)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
train = optimizer.minimize(loss, var_list=[fc8_W, fc8_b])

init = tf.global_variables_initializer()

pred = tf.arg_max(fc8, dimension=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))

# TODO: Train and evaluate the feature extraction model.
def eval(X, y):
    n = len(X)
    total_acc = 0
    total_loss = 0
    sess = tf.get_default_session()
    
    for offset in range(0, n, batch_size):
        end = offset+batch_size
        batch_X, batch_y = X[offset:end], y[offset:end]
        batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict={features: batch_X, labels: batch_y})
        total_loss += (batch_loss*len(batch_X))
        total_acc += (batch_accuracy*len(batch_X))
        
    return total_loss/n, total_acc/n

with tf.Session() as sess:
    sess.run(init)
    n = len(X_train)
    t = time.time()
    for i in range(nb_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, n, batch_size):
            end = offset + batch_size
            batch_X, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(train, feed_dict={features: batch_X, labels: batch_y})
            if offset%(batch_size*10) == 0:
                test_loss, test_acc = eval(batch_X, batch_y)
                print("Epoch done: %.3f, Batch loss: %.5f,  Batch accuracy: %.3f" % ((end / n), test_loss, test_acc))
        val_loss, val_acc = eval(X_test, y_test)
        print("Epoch", i + 1)
        print("Time: %.3f, Validation loss: %.5f,  Validation accuracy: %.3f" % ((time.time()-t), val_loss, val_acc))
        

        
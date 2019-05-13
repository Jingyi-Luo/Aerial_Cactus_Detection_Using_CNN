# Kaggle competition: Aerial Cactus Identification
# Jingyi Luo
# Padding: VALID

import math
import cv2
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.chdir("/Users/ljyi/Desktop/SYS6016/homework/homework_03")

# Step 1: Define parameters for the CNN
# Input
height = 32
width = 32
channels = 3
#n_inputs = height * width

# Parameters for TWO convolutional layers:
conv1_fmaps = 36
conv1_ksize = 3
conv1_stride = 1
conv1_pad = 'VALID'

conv2_fmaps = 72
conv2_ksize = 3
conv2_stride = 1
conv2_pad = 'VALID'

# Define a pooling layer
pool3_dropout_rate = 0.25
pool3_fmaps = conv2_fmaps

# Define a fully connected layer
n_fc1 = 128      # 144           # units for the first fully connected layer
fc1_dropout_rate = 0.5

# Output
n_outputs = 2 #10 2?

learning_rate = 0.001

tf.reset_default_graph()

# Step 2: Set up placeholders for input data
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X") # None: # of batches
#    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels]) # -1: None
    y = tf.placeholder(tf.float32, shape=[None, n_outputs], name="y")   # tf.float32
    training = tf.placeholder_with_default(False, shape=[], name='training') # placeholder for training, default value is False

# Step 3: Set up the two convolutional layers using tf.layers.conv2d          # dimension of X: 32*32*3
conv1 = tf.layers.conv2d(X, filters=conv1_fmaps, kernel_size=conv1_ksize,     # dimension (after conv1): 30*30*36
                         strides=conv1_stride, padding=conv1_pad, activation=tf.nn.relu,
                         name='conv1')
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, # dimension (after conv2): 28*28*72
                         strides=conv2_stride, padding=conv2_pad, activation=tf.nn.relu,
                         name='conv2')

# Step 4: Set up the pooling layer with dropout using tf.nn.max_pool
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID") # dimension (after pooling): 14*14*72
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps*14*14]) # will go to fully connected dense layers, so reshape to 1-d tensor
    pool3_flat_drop = tf.layers.dropout(pool3_flat, pool3_dropout_rate, training=training) # training placeholder, False.

# Step 5: Set up the fully connected layer using tf.layers.dense
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")# width of layer: n_fc1
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

# Step 6: Calculate final output from the output of the fully connected layer
# define a sigmoid function
#def fun_sigmoid(x):
#    return tf.exp(x)/(tf.exp(x)+1)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
#    y_proba = fun_sigmoid(logits)
    y_proba = tf.sigmoid(logits)

# Step 7: Define the optimizer; taking as input (learning_rate) and (loss)
# define loss function using cross-entropy
#def fun_loss(p, y):
#    a_loss = tf.reduce_mean(-1*(y*tf.log(p)+(1-p)*tf.log(1-p)))
#    return a_loss

with tf.name_scope("loss"):
#    loss = fun_loss(y_proba, y)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Step 8: Define the evaluation metric
with tf.name_scope("eval"):
    correctPrediction = tf.equal(tf.argmax(y_proba, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
#    y_int = tf.cast(y, tf.int32)
#    correct = tf.nn.in_top_k(logits, y_int, 1)   #predictions(a batch_size*classes tensor), targets(int)
#    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Step 7: Initiate
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# Step 8: Read in data
train_dir = "kaggle_data/train/"
test_dir = "kaggle_data/test/"
train_df = pd.read_csv("kaggle_data/train.csv")   # 17500*2

# read training data
x_train = []
y_train = []
images_names = train_df['id'].values  # convert dataframe to numpy array
for img_id in images_names:  # [0:100]
#    print(img_id)
    filename = train_dir + img_id
#    image_string = tf.read_file(filename) # tf.read_file: read and output the entire contents of the input filename
#    image_decoded = tf.image.decode_image(image_string)
#    image_array = plt.imread(filename)  # read an png image from a file into an array
    image_array = cv2.imread(filename)
#    image = tf.cast(image_array, tf.float32) / 255.0

    x_train.append(image_array)
    y_train.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])
x_train = np.asarray(x_train)
x_train = x_train.astype('float32')/255
y_train = np.asarray(y_train)
y_train_encoded = np.array(pd.get_dummies(y_train))


# read in test data
x_test = []
for img_id in (os.listdir(test_dir)): # [0:100]
#    print(img_id)
    filename = test_dir + img_id
#    image_string = tf.read_file(filename)
#    image_decoded = tf.image.decode_image(image_string)
    image_array = cv2.imread(filename)
#    image = tf.cast(image_array, tf.float32) / 255.0
    x_test.append(image_array)
x_test = np.asarray(x_test)
x_test = x_test.astype('float32')/255   # (4000, 32, 32, 3)

# split training data into train and validata
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train_encoded, test_size = 0.1, random_state=8)
# len(x_tr) = 15750
# len(y_tr) = 15750
# len(x_val) = 1750
# len(x_val) = 1750

# function for shuffling
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# Step 9: Define some necessary functions
def get_model_params(): # get parameters at every step of early stopping
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params): # resort parameter for that step of training.
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

# Step 10: Define training and evaluation parameters
n_epochs = 18
batch_size = 50
iteration = 0

best_loss_val = np.infty  # really large. if less that this large one, will update
check_interval = 500 #   # 20
checks_since_last_progress = 0  # how many intervals (500 steps) has been passed, keeping the progress.
max_checks_without_progress = 20 # if 20 intervals, stop
best_model_params = None

# ----------- The codes for writing graph to tensorboard ----------------------
# a writer to write the CNN graph tensorboard
writer = tf.summary.FileWriter('./graphs/computation_graph_nonpadding', tf.get_default_graph())

# For loss/accuracy with epoches on Tensorboard
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./graphs/train_padding_VALID', tf.get_default_graph()) # train_dropout_0.2 #train_layer_2
test_writer = tf.summary.FileWriter('./graphs/test_padding_VALID', tf.get_default_graph())
# -----------------------------------------------------------------------------

# Step 11: Train and evaluate CNN with Early Stopping procedure defined at the very top
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(x_tr, y_tr, batch_size):
            iteration += 1
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration%check_interval == 0: # if batch # in first epoch < 500, best_loss_val is inf.
                loss_val = loss.eval(feed_dict={X: x_val, y: y_val})# session.run.loss
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params() # save this state and retrain...???

                else:
                    checks_since_last_progress +=1
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: x_val, y: y_val})
        print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                  epoch, acc_batch * 100, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress: # if 20 times, since best_loss_val
            print("Early stopping!")
            break

# ----------- The codes for writing graph to tensorboard ----------------------
        # measure validation accuracy, and write validate summaries to FileWriters
        test_summary, acc = sess.run([merged, accuracy], feed_dict={X: x_val, y: y_val})
        test_writer.add_summary(test_summary, epoch)
        print('Accuracy at step %s: %s' % (epoch, acc))

        # run training_op on training data, and add training summaries to FileWriters
        train_summary, _ = sess.run([merged, training_op], feed_dict={X:X_batch, y:y_batch}) # , training_op
        train_writer.add_summary(train_summary, epoch)

    train_writer.close()
    test_writer.close()
# -----------------------------------------------------------------------------

    if best_model_params:
        restore_model_params(best_model_params)
    save_path = saver.save(sess, "./model_optimization/my_model_final.ckpt")
writer.close()

# ------------------------ Test set prediction --------------------------------
with tf.Session() as sess:
    saver.restore(sess, "./model_optimization/my_model_final.ckpt")
    Z = y_proba.eval(feed_dict = {X: x_test})
    y_pred = np.argmax(Z, axis = 1)
print("Predicted classes:", y_pred)

# write to a dataframe to upload to kaggle
output = pd.DataFrame(y_pred, columns=['has_cactus'])
output['id'] = os.listdir(test_dir)

# swap two columns
columnsTitles=["id","has_cactus"]
output=output.reindex(columns=columnsTitles)

# write to csv
output.to_csv("prediction_kaggle_cactus.csv", index=False)

#print(tf.__version__)

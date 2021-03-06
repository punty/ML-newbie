#import the required library

import tensorflow as tf
from enum import Enum


RUN_NAME = "LB - 512"
LOGDIR = './tmp/{}/'.format(RUN_NAME)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           200, 0.96, staircase=True)
num_empochs = 6000

#placeholder data
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
isTraing = tf.placeholder(tf.bool)

class NNModel:

    def __init__(self, input_size):
        self.layers = []
        self.input_size = input_size

    def appendLayer(self, layer_type, numberOfNodes=2):
        assert isinstance(layer_type, NNLayerActivation)
        assert isinstance(numberOfNodes, int)
        index = len(self.layers)
        numberOfInputs = self.input_size[1]
        if index > 0:
            numberOfInputs = self.layers[-1].shape[1]
        layer = NNLayer(layer_type=layer_type,numberOfNodes=numberOfNodes,numberOfInputs=numberOfInputs,index=index)
        self.layers.append(layer)

    def computeForward(self, X):
        A = X
        for layer in model.layers:
            A = layer.compute_forward(A)
        return A

class NNLayerActivation(Enum):
    LINEAR_RELU = 1
    LINEAR = 2
    LINEAR_DROPOUT = 3
    LINEAR_BATCH_RELU = 4

class NNLayer:

    def __init__(self, layer_type, numberOfNodes, numberOfInputs, index):
        assert isinstance(layer_type, NNLayerActivation)
        assert isinstance(numberOfNodes, int)
        assert isinstance(numberOfInputs, int)
        assert isinstance(index, int)
        self.type = layer_type
        self.index = index
        self.shape = [numberOfInputs,numberOfNodes]

    def compute_forward(self, X):
        with tf.variable_scope('layer_'+str(self.index)):
            W = tf.get_variable("W"+str(self.index),shape=self.shape,dtype=tf.float32,initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b"+str(self.index),shape=self.shape[1],dtype=tf.float32,initializer = tf.zeros_initializer())
            Z = tf.matmul(X,W) + b
            A = self.activation(Z)
            return A

    def activation(self, Z):
        if self.type == NNLayerActivation.LINEAR_BATCH_RELU:
            batch_mean2, batch_var2 = tf.nn.moments(Z, [0])
            scale2 = tf.Variable(tf.ones(self.shape[1]))
            beta2 = tf.Variable(tf.zeros(self.shape[1]))
            BN2 = tf.nn.batch_normalization(Z, batch_mean2, batch_var2, beta2, scale2, 0.0001)
            return tf.nn.relu(BN2)


        if self.type == NNLayerActivation.LINEAR_RELU:
            return tf.nn.relu(Z)
        if self.type == NNLayerActivation.LINEAR:
            return Z
        if self.type == NNLayerActivation.LINEAR_DROPOUT:
            a = tf.nn.dropout(Z,keep_prob)
            return tf.nn.relu(a)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

#Build Model

model = NNModel(mnist.train.images.shape)
model.appendLayer(NNLayerActivation.LINEAR_RELU,128)
model.appendLayer(NNLayerActivation.LINEAR_DROPOUT,256)
model.appendLayer(NNLayerActivation.LINEAR_BATCH_RELU,128)
model.appendLayer(NNLayerActivation.LINEAR_RELU,128)
model.appendLayer(NNLayerActivation.LINEAR,10)


#construct forward tree
prediction = model.computeForward(X)


with tf.variable_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network
with tf.variable_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('Accuracy', accuracy)
    summary = tf.summary.merge_all()

#every 10 epoch calculate e print summary of cost over the whole dataset
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    training_writer = tf.summary.FileWriter(LOGDIR+'training',session.graph)
    test_writer = tf.summary.FileWriter(LOGDIR + 'test', session.graph)
    for epoch in range (num_empochs):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _ = session.run(train_step,feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.6, isTraing: 1})
        if epoch % 10 == 0:
            acc_tr,training_cost, training_summary = session.run([accuracy,cost, summary], feed_dict={X:mnist.train.images, Y:mnist.train.labels, keep_prob:1.0, isTraing: 0})
            acc_test,test_cost, test_summary = session.run([accuracy,cost, summary], feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0, isTraing: 0})
            training_writer.add_summary(training_summary, epoch)
            test_writer.add_summary(test_summary, epoch)
            print("Epoch: {} - Training Acc: {}  Testing Acc: {}".format(epoch, acc_tr, acc_test))
    model_builder = tf.saved_model.builder.SavedModelBuilder('exported_model')
    inputs = {
     'input': tf.saved_model.utils.build_tensor_info(X)
    }

    outputs = {
        'output': tf.saved_model.utils.build_tensor_info(Y)
    }

    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    model_builder.add_meta_graph_and_variables(
    session,
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        }
    )

    model_builder.save()

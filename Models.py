"""
File: Models.py
Author: Ameya Shringi as6520@g.rit.edu
        Vishal Garg
Description: Create Models based on the test and training set
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.contrib import skflow
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import cPickle as pkl


train_data = 'train_data.csv'
train_attribute = 'train_attribute.csv'
test_data = 'test_data.csv'
test_attribute = 'test_attribute.csv'
number_of_classes = 7

class  Models:
    __slots__ = 'train_data', 'test_data', 'train_attribute', 'test_attribute'

    def __init__(self, train_data, test_data, train_attribute, test_attribute):
        """
        Constructor for the class
        :param train_data: matrix
        :param test_data:
        :param train_attribute:
        :param test_attribute:
        :return:
        """
        self.train_data = train_data
        self.train_attribute = train_attribute
        self.test_data = test_data
        self.test_attribute = test_attribute

    def linearClassifierTrain(self):
        """
        Softmax Classifier
        :return: None
        """
        classfier = skflow.TensorFlowLinearClassifier(n_classes=7, batch_size=256, steps=10000, learning_rate=0.01)
        classfier.fit(self.train_data, np.argmax(self.train_attribute, 1))
        print "Ended Fitting"
        print accuracy_score(np.argmax(self.test_attribute,1), classfier.predict(self.test_data))
        print classification_report(np.argmax(self.test_attribute, 1), classfier.predict(self.test_data))
        print confusion_matrix(np.argmax(self.test_attribute, 1), classfier.predict(self.test_data))

    def svmClassifieTrainr(self):
        """
        SVM Classifier
        :return: None
        """
        classifier = svm.SVC()
        classifier.fit(self.train_data, np.argmax(self.train_attribute, 1))
        with open("svmclassifier.pkl", 'wb') as f:
            pkl.dump(classifier, f)
        print accuracy_score(np.argmax(self.test_attribute, 1), classifier.predict(self.test_data))
        print classification_report(np.argmax(self.test_attribute, 1), classifier.predict(self.test_data))
        print confusion_matrix(np.argmax(self.test_attribute, 1), classifier.predict(self.test_data))

    def randomForest(self):
        """
        Random Forest Classifier
        :return: None
        """
        classifier = RandomForestClassifier()
        classifier.fit(self.train_data, np.argmax(self.train_attribute, 1))
        with open("randomForestclassifier.pkl", 'wb') as f:
            pkl.dump(classifier, f)
        print accuracy_score(np.argmax(self.test_attribute, 1), classifier.predict(self.test_data))
        print classification_report(np.argmax(self.test_attribute, 1), classifier.predict(self.test_data))
        print confusion_matrix(np.argmax(self.test_attribute, 1), classifier.predict(self.test_data))

    def deepNN(self):
        """
        Deep Neural Network Classifier
        :return: None
        """
        try:
            different_classifier = skflow.TensorFlowDNNClassifier.restore('dnn')
        except:
            classifier = skflow.TensorFlowDNNClassifier(hidden_units=[512, 1024, 1024, 1024, 512, 256, 64], n_classes=7, steps=20000)
            print "Fitting Data"
            classifier.fit(self.train_data, np.argmax(self.train_attribute, 1))
            classifier.save('dnn')
        print accuracy_score(np.argmax(self.test_attribute, 1), different_classifier.predict(self.test_data))
        print classification_report(np.argmax(self.test_attribute, 1), different_classifier.predict(self.test_data))
        print confusion_matrix(np.argmax(self.test_attribute, 1), different_classifier.predict(self.test_data))



    def max_pool_1D(self, tensor_in):
        """
        One Dimensional Max Pooling Function
        :param tensor_in: Input tensor vector
        :return: max-pooled layer
        """
        return tf.nn.max_pool(tensor_in, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],
                padding='SAME')

    def conv_model_1D(self, data, attribute):
        """
        Function to create one dimensional convolution model
        :param train_data:
        :param train_attribute:
        :return:
        """
        data = tf.reshape(data, [-1, 256, 2, 1])
        with tf.variable_scope('conv_layer1'):
            h_conv1 = skflow.ops.conv2d(data, n_filters=32, filter_shape=[9, 1],
                                        bias=True, activation=tf.nn.relu)
            h_pool1 = self.max_pool_1D(h_conv1)
        with tf.variable_scope('conv_layer2'):
            h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=48, filter_shape=[7, 1],
                                        bias=True, activation=tf.nn.relu)
            h_pool2 = self.max_pool_1D(h_conv2)

        with tf.variable_scope('conv_layer3'):
            h_conv3 = skflow.ops.conv2d(h_pool2, n_filters=64, filter_shape=[5, 1],
                                        bias=True, activation=tf.nn.relu)
            h_pool3 = self.max_pool_1D(h_conv3)

        with tf.variable_scope('conv_layer4'):
            h_conv4 = skflow.ops.conv2d(h_pool3, n_filters=48, filter_shape=[3, 1],
                                        bias=True, activation=tf.nn.relu)
            h_pool4 = self.max_pool_1D(h_conv4)
        with tf.variable_scope('conv_layer5'):
            h_conv5 = skflow.ops.conv2d(h_pool4, n_filters=32, filter_shape=[3, 1],
                                        bias=True, activation=tf.nn.relu)
            h_pool5 = self.max_pool_1D(h_conv5)
            h_pool5_flat = tf.reshape(h_pool5, [-1, 8 * 2 * 32])
        h_fc1 = skflow.ops.dnn(h_pool5_flat, [1024], activation=tf.nn.relu)
        return skflow.models.logistic_regression(h_fc1, attribute)

    def convNetRectangular(self):
        """
        One Dimensional Convolution Network
        :return: None
        """
        # Training and predicting
        classifier = skflow.TensorFlowEstimator(
            model_fn=self.conv_model_1D, n_classes=7,
            batch_size=64, steps=20000, learning_rate=0.005)

        classifier.fit(self.train_data, np.argmax(self.train_attribute, 1))
        classifier.save("conv_net")
        print accuracy_score(np.argmax(self.test_attribute, 1),
                             classifier.predict(self.test_data))
        print classification_report(np.argmax(self.test_attribute, 1),
                                    classifier.predict(self.test_data))
        print confusion_matrix(np.argmax(self.test_attribute, 1),
                               classifier.predict(self.test_data))


def main():
    """
    Main Function
    :return: None
    """
    train_data_val = np.loadtxt(train_data)
    train_attribute_val = np.loadtxt(train_attribute)
    test_data_val = np.loadtxt(test_data)
    test_attribute_val = np.loadtxt(test_attribute)
    s = Models(train_data_val, test_data_val, train_attribute_val, test_attribute_val)
    s.linearClassifierTrain()
    s.svmClassifieTrainr()
    s.randomForest()
    s.deepNN()
    s.convNetRectangular()


if __name__ == '__main__':
    main()


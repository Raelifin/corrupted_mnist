# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

# from tensorflow.examples.tutorials.mnist import input_data
# pure_mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

import tensorflow as tf
import numpy as np

FLAGS = None


def one_hot(hot_index, length):
  return [1.0 if hot_index==i else 0.0 for i in range(length)]

def import_data(filepath):
  print("Reading data from %s..." % filepath)
  xs = []
  ys = []
  with open(filepath) as datafile:
    for line in datafile:
      values = [float(x) for x in line.split(',')]
      xs.append(values[1:])
      ys.append(one_hot(values[0], 10))
  return xs, ys

def next_batch(data, start_location, samples):
  # Use modulus to redraw from the beginning of the data if the start_location is too high.
  start_location = start_location % len(data)
  if start_location+samples > len(data):
    # Concatinate a copy of the data onto the end to replicate redrawing from the beginning.
    data = data + data
    # This isn't very memory efficient, but I'm hacking a solution together. FIXME
  return np.array(data[start_location:start_location+samples])

def main(_):
  # Import data
  images, labels = import_data(FLAGS.input_data)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train
  print("Starting training...")
  for i in range(1000):
    batch_xs = next_batch(images, i*100, 100)
    batch_ys = next_batch(labels, i*100, 100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  print("Training complete! Checking accuracy...")
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  score = sess.run(accuracy, feed_dict={x: np.array(images), y_: np.array(labels)})
  print("%2.2f%% of samples correctly labeled." % (100*score))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_data', type=str, default='data.csv', help='Input data file')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

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
CORRUPTION_RATIO = 0.1


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
  print("Training complete!")

  # Calibrating
  print("Calibrating skip ratio based on network confidence...")
  # TODO Currently I'm running a tf session to get the confidences and then computing the threshold in python.
  # Ideally the confidence threshold would be drawn directly from the network and we wouldn't need to re-run the session.
  conf_node = tf.reduce_max(tf.nn.softmax(y, 1), 1)
  confidences = sess.run(conf_node, feed_dict={x: np.array(images), y_: np.array(labels)})
  sorted_conf = np.sort(confidences)
  skip_conf = sorted_conf[int(round(len(sorted_conf)*CORRUPTION_RATIO))]
  print("Will skip on inputs where less than %2.2f%% confident." % (100*skip_conf))

  # Test trained model
  print("Testing model....")  # TODO Get validation dataset so we're not testing on training data!
  not_skip = tf.greater(tf.reduce_max(tf.nn.softmax(y, 1), 1), skip_conf)
  predictions = tf.boolean_mask(tf.argmax(y, 1), not_skip)
  truth_of_predicted_inputs = tf.boolean_mask(tf.argmax(y_, 1), not_skip)
  correct = tf.equal(predictions, truth_of_predicted_inputs)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  out = sess.run(accuracy, feed_dict={x: np.array(images), y_: np.array(labels)})
  print("%2.2f%% of samples correctly labeled." % (100*out))

  if FLAGS.save_dir:
      # Save model to disk
      saver = tf.train.Saver()
      saver.save(sess, FLAGS.save_dir+'/model.ckpt')
      print("Model saved to disk in %s" % FLAGS.save_dir)

      tf.summary.scalar('cross_entropy', cross_entropy)
      writer = tf.summary.FileWriter(FLAGS.save_dir, sess.graph)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_data', type=str, default='data.csv', help='Input data file')
  parser.add_argument('--save_dir', type=str, default=None, help='Input data file')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

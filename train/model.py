#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import warnings

import numpy as np
import tensorflow as tf


def add_binary_layer(prev_layer, num_outputs, inference, uid):
    """Adds a binary layer to the network.

    Args:
        prev_layer:    2D Tensor with shape (batch_size, num_hidden).
        num_outputs:   int, number of dimensions in next hidden layer.
        inference:     bool scalar Tensor, tells the network if it is
                       the inference step.
        uid:           int, the unique identifier.

    Returns:
        tuple (activation, updates)
            activation:     output of this layer.
            updates:        variables used to compute gradients.
    """

    batch_size, num_inputs = [dim.value for dim in prev_layer.get_shape()]

    def _pos_neg_like(tensor):
        return (tf.constant(1, dtype=tf.float32, shape=tensor.get_shape()),
                tf.constant(-1, dtype=tf.float32, shape=tensor.get_shape()))

    with tf.variable_scope('binary_layer_%d' % uid) as vs:
        w = tf.get_variable('w', shape=(num_inputs, num_outputs),
                            initializer=tf.random_normal_initializer(seed=uid),
                            trainable=True)

        # Binarize the weight matrix.
        bin_w = tf.select(w > 0, *_pos_neg_like(w))

        # Output has shape (batch_size, num_outputs)
        wx = tf.matmul(prev_layer, bin_w)

        with tf.variable_scope('activation'):
            p = tf.tanh(wx)

            def _binomial_activation():
                sigma = tf.random_uniform(
                    p.get_shape(), seed=uid) < (p + 1) / 2
                return tf.select(sigma, *_pos_neg_like(p))

            def _binary_activation():
                return tf.select(wx > 0, *_pos_neg_like(wx))

            activation = tf.cond(inference,
                                 _binomial_activation,  # Stochastic inference.
                                 _binary_activation)  # Deterministic inference.

        return activation, (p, prev_layer, w, bin_w)


def build_model(input_var, layers=[]):
    """Builds a BNN model.

    Args:
        input_var:     2D Tensor with shape (batch_size, num_input_features).
        layers:        list of ints, the dimensionality of each layer.

    Returns:
        tuple (output_layer, updates, inference)
            output_layer:    2D Tensor, the output of the model.
            updates:         list of variables used to compute gradients.
            inference:       scalar bool Tensor, used to tell the model when to
                             use inference mode.
    """

    inference = tf.placeholder(dtype=tf.bool, name='inference')
    updates = []
    hidden_layer = input_var
    for i, num_hidden in enumerate(layers):
        if num_hidden % 16 != 0:
            warnings.warn('Hidden layers should be multiples of '
                          '16, not %d' % num_hidden)

        hidden_layer, update = add_binary_layer(hidden_layer, num_hidden,
                                                inference, i)
        updates.append(update)
    output_layer = hidden_layer

    return output_layer, updates, inference


def get_loss(output, target):
    return tf.reduce_mean(tf.square(output - target))


def get_accuracy(output, target, element_wise=True):
    if not element_wise:
        output, target = tf.reduce_sum(output, [1]), tf.reduce_sum(target, [1])
    eq = tf.equal(tf.greater(output, 0), tf.greater(target, 0))
    return tf.reduce_mean(tf.cast(eq, tf.float32))


def binary_backprop(loss, output, updates):
    """Manually backpropagates gradient error.

    Args:
        loss:         scalar Tensor, the model loss.
        output:       scalar
        updates:      list of updates to use for backprop.

    Returns:
        backprop_updates:  list of (grad, variable) tuples, the gradients.
    """

    backprop_updates = []
    loss_grad, = tf.gradients(loss, output)

    for p, prev_layer, w, bin_w in updates[::-1]:
        w_grad, loss_grad = tf.gradients(p, [bin_w, prev_layer], loss_grad)
        backprop_updates.append((w_grad, w))

    return backprop_updates


def save_model(path, binary_weights):
    with open(os.path.join(path, 'model.def'), 'w') as f:
        f.write('bnn')
        for i, weight in enumerate(binary_weights):
            weight = weight.T
            f.write('\n%d,%d\n' % (weight.shape[1], weight.shape[0]))
            bstr = '\n'.join(''.join('0' if e < 0 else '1' for e in r)
                             for r in weight)
            f.write(bstr)
        f.write('\n0,0')


def main():
    parser = argparse.ArgumentParser(
        description='Train binary neural network weights.')
    parser.add_argument(
        '--save_path',
        type=str,
        help='Where to save the weights and configuration.',
        required=True)
    parser.add_argument(
        '--num_train',
        type=int,
        help='Number of iterations to train model.',
        default=20000)
    parser.add_argument(
        '--eval_every',
        type=int,
        help='How often to evalute the model.',
        default=1000)
    args = parser.parse_args()

    input_var = tf.placeholder(dtype=tf.float32, shape=(4, 32),
                               name='input_placeholder')
    output_var = tf.placeholder(dtype=tf.float32, shape=(4, 32),
                                name='output_placeholder')
    layer_sizes = [64, 64, 32]

    # Configures data for a simple XOR task.
    input_data = np.zeros(shape=(4, 32))
    input_data[(0, 0, 1, 2), (0, 1, 1, 0)] = 1
    output_data = np.zeros(shape=(4, 32))
    output_data[(0, 3), :] = 1
    input_data = input_data * 2 - 1
    output_data = output_data * 2 - 1

    feed_dict = {
        input_var: input_data,
        output_var: output_data,
    }

    model_scope = 'model'
    with tf.variable_scope(model_scope):
        output_layer, updates, inference = build_model(
            input_var, layers=layer_sizes)
        model_vars = tf.get_collection(tf.GraphKeys.VARIABLES,
                                       scope=model_scope)
        loss = get_loss(output_layer, output_var)
        accuracy = get_accuracy(output_layer, output_var)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = 0.01

    gradients = binary_backprop(loss, output_layer, updates)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    min_op = optimizer.apply_gradients(gradients, global_step=global_step)

    # Get model weights.
    _, _, _, weights = zip(*updates)

    best_accuracy = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(1, args.num_train + 1):
            feed_dict[inference] = True
            sess.run(min_op, feed_dict=feed_dict)
            if i % args.eval_every == 0:
                feed_dict[inference] = False
                accuracy_val, loss_val = sess.run([accuracy, loss],
                                                  feed_dict=feed_dict)
                print('Epoch = %d: Loss = %.4f, Accuracy = %.4f' %
                      (i, loss_val, accuracy_val))
                if accuracy_val > best_accuracy:
                    for entry in sess.run(output_layer, feed_dict=feed_dict):
                        out_str = ''.join('0' if i < 0 else '1' for i in entry)
                        print(out_str[::-1])
                    save_model(args.save_path, sess.run(weights))
                    best_accuracy = accuracy_val


if __name__ == '__main__':
    main()

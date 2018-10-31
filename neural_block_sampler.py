import numpy as np

import tensorflow as tf


class NeuralBlockSampler(object):

    def __init__(self, input_dim, hidden_dim, num_mixtures, proposal_variables, learning_rate):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.proposal_variables = proposal_variables
        self.learning_rate = learning_rate

        self.input_ph = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32)
        self.labels_ph = tf.placeholder(shape=[None, self.proposal_variables], dtype=tf.float32)

        self.layer_1 = tf.layers.dense(self.input_ph, self.hidden_dim, activation=tf.nn.relu)
        self.layer_2 = tf.layers.dense(self.layer_1, self.hidden_dim, activation=tf.nn.relu)

        self.output_weights = tf.layers.dense(self.layer_2, self.num_mixtures, activation=tf.nn.softmax)
        self.output_weights = tf.reshape(self.output_weights, shape=[-1, self.num_mixtures, 1])
        self.output_probs_matrix = tf.reshape(tf.layers.dense(self.layer_2,
                                                   self.proposal_variables * self.num_mixtures,
                                                   activation=tf.nn.sigmoid), shape=[-1, self.proposal_variables, self.num_mixtures])

        self.output = tf.squeeze(tf.matmul(self.output_probs_matrix, self.output_weights))

        self.loss = -tf.reduce_mean(self.labels_ph * tf.log(self.output + 1e-6) + (1 - self.labels_ph) * tf.log((1 - self.output + 1e-6)))
        self.loss_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def train_step(self, inputs, labels):
        loss, _, output, weights = self.sess.run([self.loss, self.loss_optimizer, self.output, self.output_weights], feed_dict={
            self.input_ph: inputs,
            self.labels_ph: labels
        })

        return loss

    def sample_step(self, motif, block, sample):
        input = []
        for motif_node in motif.motif_nodes:
            node = block[motif_node]
            cpt = motif.probs[node]
            for _, entry in cpt.items():
                input.append(entry)

        for motif_node in motif.nodes:
            if motif_node in motif.motif_evidence_vars:
                input.append(sample[block[motif_node]])

        input = np.array(input)
        input = np.reshape(input, (1, len(input)))

        output = self.sess.run([self.output],
                      feed_dict={
                          self.input_ph: input
                      })
        return output










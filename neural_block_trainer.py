import argparse
import pickle
import random

import numpy as np
import tensorflow as tf

import math
from neural_block_sampler import NeuralBlockSampler
from neural_block_motif import NeuralBlockMotif


class NeuralBlockTrainer(object):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_mixtures,
                 proposal_variables,
                 learning_rate,
                 training_epochs,
                 testing_epochs,
                 training_batch_size,
                 checkpoint_name,
                 restore,
                 restore_epoch):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.proposal_variables = proposal_variables
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.testing_epochs = testing_epochs
        self.training_batch_size = training_batch_size
        self.checkpoint_name = checkpoint_name
        self.restore = restore
        self.restore_epoch = restore_epoch

        self.model = NeuralBlockSampler(input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        num_mixtures=num_mixtures,
                                        proposal_variables=proposal_variables,
                                        learning_rate=learning_rate)

    def train_policy(self, dataset):
        start_epoch = 0
        if self.restore:
            self.model.saver.restore(self.model.sess, "checkpoints/{}/{}.ckpt".format(self.checkpoint_name, self.restore_epoch))
            start_epoch = self.restore_epoch + 1

        losses = []
        epochs = []
        for epoch in range(self.training_epochs):
            for inputs, labels in self.dataset_iterator(dataset):
                loss = self.model.train_step(inputs, labels)
                losses.append(loss)
                epochs.append(start_epoch + epoch)
            if (start_epoch + epoch) % 10 == 0:
                print("Epoch {} Loss {}".format(start_epoch + epoch, losses[-1]))
                save_path = self.model.saver.save(self.model.sess, "checkpoints/{}/{}.ckpt".format(self.checkpoint_name, start_epoch + epoch))

        data = {
            'epochs': epochs,
            'losses': losses
        }
        pickle.dump(data, open("training/{}_losses.pkl".format(self.checkpoint_name), "wb"))

    def dataset_iterator(self, dataset):
        inputs, labels = dataset['inputs'], dataset['labels']

        all_indices = np.array([i for i in range(len(inputs))])
        np.random.shuffle(all_indices)

        i = 0
        while i < len(inputs):
            indices = all_indices[i:i + self.training_batch_size]

            yield inputs[indices], labels[indices]
            i += self.training_batch_size

    def test_policy(self, model_name, exp_name, restore_epoch, gibbs):
        self.model.saver.restore(self.model.sess, "checkpoints/{}/{}.ckpt".format(self.checkpoint_name, restore_epoch))

        motif_data_path = "data/{}_motif.pkl".format(exp_name)
        motif_data = pickle.load(open(motif_data_path, 'rb'))

        model_path = "evaluation/pickle/{}.pkl".format(model_name)
        data = pickle.load(open(model_path, 'rb'))

        motif = NeuralBlockMotif(motif_data['nodes'],
                                 motif_data['edges'],
                                 motif_data['parents'],
                                 motif_data['evidence_vars'],
                                 motif_data['sampling_var'],
                                 data['nodes'],
                                 data['edges'],
                                 data['parents'],
                                 data['evidence'],
                                 data['probs'],
                                 data['marginals'])

        sample = motif.prior_sample()
        losses = []
        all_samples = []
        alphas = []
        p_ratios = []
        q_ratios = []
        p_acceptance = []

        new_order = [i for i in range(len(motif.nodes))]

        for epoch in range(self.testing_epochs):
            epoch_samples = []
            for node in new_order:
                if node in motif.evidence_vars:
                    continue
                block = motif.blocks[node]
                if gibbs or block is None:
                    sample = motif.gibbs_sample(node, sample)
                else:

                    output = self.model.sample_step(motif, block, sample)
                    candidate_sample = motif.neural_block_sample(output, block, sample)

                    # reverse_output = self.model.sample_step(motif, block, candidate_sample)
                    # print("Output", output)
                    # print("Reverse output", reverse_output)

                    q_prev_given_cand = motif.cond_prob_sample(output, block, sample)
                    q_cand_given_prev = motif.cond_prob_sample(output, block, candidate_sample)
                    p_cand = motif.joint_prob_sample_smaller(candidate_sample, block)
                    p_prev = motif.joint_prob_sample_smaller(sample, block)

                    # alpha = min(1.0, math.exp(q_prev_given_cand + p_cand - q_cand_given_prev - p_prev))
                    alpha = min(1.0, math.exp(q_prev_given_cand + p_cand - q_cand_given_prev - p_prev))
                    alpha *= 1000.0
                    alphas.append(alpha)
                    p_ratios.append(math.exp(p_cand - p_prev))
                    q_ratios.append(math.exp(q_prev_given_cand - q_cand_given_prev))

                    # print("P cand {} P prev {}".format(math.exp(p_cand), math.exp(p_prev)))
                    # print("P ratio", math.exp(p_cand - p_prev))
                    # print("Q ratio", math.exp(q_prev_given_cand - q_cand_given_prev))
                    # print("Alpha", alpha)

                    p = random.random()

                    if p <= 0.4:
                        sample = candidate_sample
                        p_acceptance.append(1)
                    else:
                        sample = sample.copy()
                        p_acceptance.append(0)
                all_samples.append(sample)
                epoch_samples.append(sample)
            loss = motif.evaluate_samples(all_samples)
            losses.append(loss)
            if epoch % 10 == 0:
                print("Epoch {} Loss {}".format(epoch, loss))

        loss = motif.evaluate_samples(all_samples)
        print("Total loss {}".format(loss))
        print("Mean alpha {}".format(np.mean(alphas)))
        print("Mean p {}".format(np.mean(p_ratios)))
        print("Mean q {}".format(np.mean(q_ratios)))
        print("Mean acceptance {}".format(np.mean(p_acceptance)))
        pickle.dump(losses, open("testing/{}_{}losses.pkl".format(self.checkpoint_name, 'gibbs_' if gibbs else ''), "wb"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=106)
    parser.add_argument('--hidden_dim', type=int, default=480)
    parser.add_argument('--num_mixtures', type=int, default=12)
    parser.add_argument('--proposal_variables', type=int, default=9)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--training_epochs', type=int, default=1000)
    parser.add_argument('--training_batch_size', type=int, default=1024)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='grid')
    parser.add_argument('--checkpoint_name', type=str, default='grid')
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--restore_epoch", type=int, default=990)
    parser.add_argument('--model_name', type=str, default='50-19-4')
    parser.add_argument('--testing_epochs', type=int, default=500)
    parser.add_argument("--gibbs", action="store_true")

    args = parser.parse_args()

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    trainer = NeuralBlockTrainer(input_dim=args.input_dim,
                                 hidden_dim=args.hidden_dim,
                                 num_mixtures=args.num_mixtures,
                                 proposal_variables=args.proposal_variables,
                                 learning_rate=args.learning_rate,
                                 training_epochs=args.training_epochs,
                                 testing_epochs=args.testing_epochs,
                                 training_batch_size=args.training_batch_size,
                                 checkpoint_name=args.checkpoint_name,
                                 restore=args.restore,
                                 restore_epoch=args.restore_epoch)

    dataset = pickle.load(open("data/{}.pkl".format(args.exp_name), "rb"))
    if args.test:
        trainer.test_policy(args.model_name, args.exp_name, args.restore_epoch, args.gibbs)
    else:
        trainer.train_policy(dataset)


if __name__ == "__main__":
    main()

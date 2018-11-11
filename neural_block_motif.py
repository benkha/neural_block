import math

import numpy as np
import random

class NeuralBlockMotif(object):

    def __init__(self, motif_nodes, motif_edges, motif_parents, motif_evidence_vars, motif_sampling_var, nodes, edges, parents, evidence, probs, marginals):
        self.motif_nodes = motif_nodes
        self.motif_edges = motif_edges
        self.motif_parents = motif_parents
        self.motif_evidence_vars = set(motif_evidence_vars)
        self.motif_sampling_var = motif_sampling_var

        self.nodes = nodes
        self.edges = edges
        self.parents = parents
        self.evidence = evidence
        self.evidence_vars = evidence.keys()
        self.probs = probs
        self.marginals = marginals

        self.blocks = self.find_blocks()

    def find_block(self, sampling_var):
        motif_mapping = {}

        node_11 = sampling_var
        if self.check_parents(node_11) or self.check_edges(node_11):
            return None

        node_6, node_7 = self.parents[sampling_var]
        if self.check_parents(node_6) or self.check_edges(node_6):
            return None
        if self.check_parents(node_7) or self.check_edges(node_7):
            return None

        node_3 = self.find_common(node_6, node_7)
        if self.check_parents(node_3) or self.check_edges(node_3):
            return None

        node_0, node_1 = self.parents[node_3]
        if self.check_parents(node_0) or self.check_edges(node_0):
            return None
        if self.check_parents(node_1) or self.check_edges(node_1):
            return None

        node_2 = self.set_subtraction(self.parents[node_6], node_3)
        if self.check_parents(node_2) or self.check_edges(node_2):
            return None

        node_4 = self.set_subtraction(self.parents[node_7], node_3)
        if self.check_parents(node_4) or self.check_edges(node_4):
            return None

        # print("Node 11", node_11)
        # print("Node 6", node_6)
        # print("Node 7", node_7)
        # print("Node 3", node_3)
        # print("Node 0", node_0)
        # print("Node 1", node_1)
        # print("Node 2", node_2)
        # print("Node 4", node_4)

        node_10 = self.set_subtraction(self.edges[node_6], node_11)
        if self.check_parents(node_10) or self.check_edges(node_10):
            return None

        node_12 = self.set_subtraction(self.edges[node_7], node_11)
        if self.check_parents(node_12) or self.check_edges(node_12):
            return None

        node_5 = self.set_subtraction(self.parents[node_10], node_6)
        if self.check_parents(node_5) or self.check_edges(node_5):
            return None

        node_8 = self.set_subtraction(self.parents[node_12], node_7)
        if self.check_parents(node_8) or self.check_edges(node_8):
            return None

        node_9 = self.set_subtraction(self.edges[node_5], node_10)
        if self.check_parents(node_9):
            return None

        node_13 = self.set_subtraction(self.edges[node_8], node_12)
        if self.check_parents(node_13):
            return None

        node_15 = self.find_common(node_10, node_11, parents=False)
        if self.check_parents(node_15) or self.check_edges(node_15):
            return None

        node_16 = self.find_common(node_12, node_11, parents=False)
        if self.check_parents(node_16) or self.check_edges(node_16):
            return None

        node_14 = self.set_subtraction(self.edges[node_10], node_15)
        if self.check_parents(node_14):
            return None

        node_17 = self.set_subtraction(self.edges[node_12], node_16)
        if self.check_parents(node_17):
            return None

        node_19 = self.find_common(node_15, node_16, parents=False)
        if self.check_parents(node_19) or self.check_edges(node_19):
            return None

        node_18 = self.set_subtraction(self.edges[node_15], node_19)
        if self.check_parents(node_18):
            return None

        node_20 = self.set_subtraction(self.edges[node_16], node_19)
        if self.check_parents(node_20):
            return None

        node_21 = self.find_common(node_18, node_19, parents=False)
        if self.check_parents(node_21):
            return None

        node_22 = self.find_common(node_20, node_19, parents=False)
        if self.check_parents(node_22):
            return None

        motif_mapping[0] = node_0
        motif_mapping[1] = node_1
        motif_mapping[2] = node_2
        motif_mapping[3] = node_3
        motif_mapping[4] = node_4
        motif_mapping[5] = node_5
        motif_mapping[6] = node_6
        motif_mapping[7] = node_7
        motif_mapping[8] = node_8
        motif_mapping[9] = node_9
        motif_mapping[10] = node_10
        motif_mapping[11] = node_11
        motif_mapping[12] = node_12
        motif_mapping[13] = node_13
        motif_mapping[14] = node_14
        motif_mapping[15] = node_15
        motif_mapping[16] = node_16
        motif_mapping[17] = node_17
        motif_mapping[18] = node_18
        motif_mapping[19] = node_19
        motif_mapping[20] = node_20
        motif_mapping[21] = node_21
        motif_mapping[22] = node_22

        for motif_node, node in motif_mapping.items():
            if motif_node not in self.motif_evidence_vars and node in self.evidence_vars:
                return None

        return motif_mapping

    def check_parents(self, node):
        return node not in self.parents or len(self.parents[node]) != 2

    def check_edges(self, node):
        return node not in self.edges or len(self.edges[node]) != 2

    def set_subtraction(self, lst1, node):
        temp_set = set(lst1) - set([node])
        assert(len(temp_set) == 1)

        return temp_set.pop()

    def find_common(self, node_1, node_2, parents=True):
        if parents:
            common_parents = list(set(self.parents[node_1]) & set(self.parents[node_2]))
            assert(len(common_parents) == 1)
            return common_parents[0]
        else:
            common_children = list(set(self.edges[node_1]) & set(self.edges[node_2]))
            assert(len(common_children) == 1)
            return common_children[0]

    def find_blocks(self):
        blocks = {}
        for node in self.nodes:
            blocks[node] = self.find_block(sampling_var=node)
        return blocks

    def prior_sample(self):
        sample = {}
        for node in self.nodes:
            if node in self.evidence_vars:
                sample[node] = self.evidence[node]
            else:
                if node not in self.parents:
                    conditional = ()
                else:
                    conditional = []
                    for parent in self.parents[node]:
                        assert(parent in sample)
                        conditional.append(parent)
                        conditional.append(sample[parent])
                    conditional = tuple(conditional)

                assert(conditional in self.probs[node])
                entry = self.probs[node][conditional]
                p = random.random()
                if p <= entry:
                    sample[node] = 1
                else:
                    sample[node] = 0
        return sample

    def gibbs_sample(self, node, sample):
        sample = sample.copy()

        conditional = self.get_conditional(node, sample)

        entry = self.probs[node][conditional]
        entry_0 = 1.0 - entry

        for child in self.edges[node]:

            conditional = self.gibbs_conditional(child, node, 1, sample)
            conditional_0 = self.gibbs_conditional(child, node, 0, sample)

            e = self.probs[child][conditional]
            f = self.probs[child][conditional_0]

            if sample[child] == 1.0:
                entry *= e
                entry_0 *= f
            else:
                entry *= 1 - e
                entry_0 *= 1 - f

        print("node", node)
        print("sample", sample)
        assert(entry + entry_0 != 0.0)

        threshold = entry / (entry + entry_0)

        p = random.random()

        if p <= threshold:
            sample[node] = 1
        else:
            sample[node] = 0
        return sample

    def gibbs_conditional(self, node, parent, value, sample):
        parents = self.parents[node]
        assert(parent in parents)
        if len(parents) == 1:
            conditional = (parent, value)
        else:
            if parents[0] == parent:
                conditional = (parent, value, parents[1], sample[parents[1]])
            else:
                assert(parent == parents[1])
                conditional = (parents[0], sample[parents[0]], parent, value)
        return conditional



    def get_conditional(self, node, sample):
        parents = self.parents[node]
        if node not in self.parents or len(parents) == 0:
            conditional = ()
        elif len(parents) == 1:
            conditional = (parents[0], sample[parents[0]])
        else:
            conditional = (parents[0], sample[parents[0]], parents[1], sample[parents[1]])
        return conditional

    def neural_block_sample(self, output, block, sample):
        sample = sample.copy()
        i = 0
        for motif_node in self.motif_nodes:
            if motif_node not in self.motif_evidence_vars:
                node = block[motif_node]
                entry = output[0][i]
                p = random.random()

                if p <= entry:
                    sample[node] = 1.0
                else:
                    sample[node] = 0.0
                i += 1
        assert(i == len(output[0]))
        return sample

    def evaluate_samples(self, samples):
        loss = 0.0
        for node in self.nodes:
            mar_prob = self.marginals[node]
            p_hat = np.mean([sample[node] for sample in samples])
            loss += abs(mar_prob - p_hat)
        return loss / len(self.nodes)

    def cond_prob_sample(self, output, block, sample):
        log_p = 0.0
        i = 0
        for motif_node in self.motif_nodes:
            if motif_node not in self.motif_evidence_vars:
                node = block[motif_node]
                if sample[node] == 1.0:
                    log_p += math.log(output[0][i] + 1e-6)
                else:
                    assert(sample[node] == 0.0)
                    log_p += math.log(1 - output[0][i] + 1e-6)
                i += 1
        return log_p

    def joint_prob_sample(self, sample):
        log_p = 0.0
        for node in self.nodes:
            conditional = self.get_conditional(node, sample)
            entry = self.probs[node][conditional]
            if sample[node] == 1.0:
                log_p += math.log(entry + 1e-6)
            else:
                log_p += math.log(1 - entry + 1e-6)
        return log_p

    def joint_prob_sample_smaller(self, sample, block):
        log_p = 0.0
        for motif_node in self.motif_nodes:
            node = block[motif_node]
            conditional = self.get_conditional(node, sample)
            entry = self.probs[node][conditional]
            if sample[node] == 1.0:
                log_p += math.log(entry + 1e-6)
            else:
                log_p += math.log(1 - entry + 1e-6)
        return log_p

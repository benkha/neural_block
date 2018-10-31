import argparse
import os
import pickle
import random
from collections import defaultdict, OrderedDict

import numpy as np


def topological_sort(nodes, edges):
    stack = []
    visited = set()
    for node in nodes:
        if node not in visited:
            visit(node, stack, visited, edges)
    stack.reverse()
    return stack


def visit(node, stack, visited, edges):
    visited.add(node)
    if node in edges:
        for child in edges[node]:
            if child not in visited:
                visit(child, stack, visited, edges)

    stack.append(node)


class GridDataset(object):

    def __init__(self,
                 num_instantiations=100,
                 num_samples=100,
                 seed=1,
                 exp_name='grid'):
        self.nodes = [i for i in range(23)]
        self.conditioning_nodes = [
            0,
            1,
            2,
            4,
            5,
            8,
            9,
            13,
            14,
            17,
            18,
            20,
            21,
            22
        ]
        self.sampling_var = 11
        self.conditioning_nodes_set = set(self.conditioning_nodes)
        self.edges = {
            0: [2, 3],
            1: [3, 4],
            2: [5, 6],
            3: [6, 7],
            4: [7, 8],
            5: [9, 10],
            6: [10, 11],
            7: [11, 12],
            8: [12, 13],
            9: [14],
            10: [14, 15],
            11: [15, 16],
            12: [16, 17],
            13: [17],
            14: [18],
            15: [18, 19],
            16: [19, 20],
            17: [20],
            18: [21],
            19: [21, 22],
            20: [22]
        }
        self.parents = self.generate_parents()
        self.num_instantiations = num_instantiations
        self.num_samples = num_samples
        self.exp_name = exp_name
        self.nodes = topological_sort(self.nodes, self.edges)


        random.seed(seed)
        np.random.seed(seed)

        self.instantiations = []
        self.samples = defaultdict(list)

    def generate_parents(self):
        parents = defaultdict(list)
        for node in self.edges:
            if node in self.edges:
                for child in self.edges[node]:
                    parents[child].append(node)
        return parents

    def generate_instantiations(self):
        for i in range(self.num_instantiations):
            instantiation = {}
            for node in self.nodes:
                node_parents = self.parents[node]
                if len(node_parents) == 0:
                    node_parents.append(-2)
                    node_parents.append(-1)
                elif len(node_parents) == 1:
                    node_parents.append(-1)
                instantiation[node] = GridDataset.generate_cpt(node_parents)
            self.instantiations.append(instantiation)

    @staticmethod
    def generate_cpt(node_parents):
        cpt = OrderedDict()
        for i in range(2):
            for j in range(2):
                conditional = (node_parents[0], i, node_parents[1], j)
                p = random.random()
                if p <= 1.0 / 40:
                    entry = 1.0
                elif p <= 2.0 / 40:
                    entry = 0.0
                else:
                    entry = random.random()
                cpt[conditional] = entry
        return cpt

    def generate_samples(self):
        for i, instantiation in enumerate(self.instantiations):
            for _ in range(self.num_samples):
                sample = {}
                for node in self.nodes:
                    node_parents = self.parents[node]
                    cpt = instantiation[node]
                    conditional = (
                        node_parents[0],
                        sample[node_parents[0]] if node_parents[0] in sample else random.randint(0, 1),
                        node_parents[1],
                        sample[node_parents[1]] if node_parents[1] in sample else random.randint(0, 1),
                    )
                    entry = cpt[conditional]
                    p = random.random()
                    if p <= entry:
                        sample[node] = 1
                    else:
                        sample[node] = 0
                self.samples[i].append(sample)

    def save_instantiations(self):
        inputs = []
        labels = []
        for i, instantiation in enumerate(self.instantiations):
            input = []

            for node in self.nodes:
                cpt = instantiation[node]
                for _, entry in cpt.items():
                    input.append(entry)
            for sample in self.samples[i]:
                input_copy = input[:]
                label = []
                for node in self.nodes:
                    if node in self.conditioning_nodes_set:
                        input_copy.append(sample[node])
                    else:
                        label.append(sample[node])
                inputs.append(input_copy)
                labels.append(label)

        data = {
            'inputs': np.array(inputs),
            'labels': np.array(labels)
        }

        if not(os.path.exists('data')):
            os.makedirs('data')

        pickle.dump(data, open("data/{}.pkl".format(self.exp_name), "wb"))

        print("Input shape", data['inputs'].shape)
        print("Input", data['inputs'])
        print("Label shape", data['labels'].shape)
        print("Label", data['labels'])

    def save_motif(self):
        edges = defaultdict(list, self.edges)
        data = {
            'nodes': self.nodes,
            'edges': edges,
            'parents': self.parents,
            'evidence_vars': self.conditioning_nodes,
            'sampling_var': self.sampling_var
        }

        pickle.dump(data, open("data/{}_motif.pkl".format(self.exp_name), "wb"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_inst', type=int, default=10000)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default='grid')

    args = parser.parse_args()

    dataset = GridDataset(num_instantiations=args.n_inst,
                          num_samples=args.n_samples,
                          seed=args.seed,
                          exp_name=args.exp_name)

    dataset.save_motif()

    dataset.generate_instantiations()
    dataset.generate_samples()

    dataset.save_instantiations()


if __name__ == "__main__":
    main()

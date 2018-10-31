import argparse
import os
import pickle
from collections import defaultdict, OrderedDict
from grid_dataset import topological_sort

eight_var_map = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1)
}


def read_model(model):
    model_path = 'evaluation/networks/uai/{}.uai'.format(model)
    with open(model_path, 'r') as f:
        network_type = f.readline()
        assert(network_type == 'BAYES\n')

        n = int(f.readline())

        f.readline() #Should all be 2s for binary networks

        assert(int(f.readline()) == n)

        parents = {}
        factor_list = []
        probs = defaultdict(OrderedDict)

        for line in f:
            if line == '\n':
                break
            arr = line.split()

            child = int(arr[-1])

            parents[child] = [int(arr[i]) for i in range(1, len(arr) - 1)]

            factor_list.append((child, *parents[child]))

        factor_index = -1
        num_entries = None
        entry_index = None

        for line in f:
            if line == '\n':
                continue
            arr = line.split()
            if len(arr) == 1:
                factor_index += 1
                entry_index = 0
                num_entries = int(arr[0])
                continue
            else:
                child, *parents_list = factor_list[factor_index]
                conditional = get_conditional(num_entries, entry_index, parents_list)
                probs[child][conditional] = float(arr[1])
                entry_index += 1

        return n, parents, probs


def get_conditional(num_entries, entry_index, parents_list):
    if num_entries == 2:
        return ()
    if num_entries == 4:
        return parents_list[0], entry_index
    i, j = eight_var_map[entry_index]
    return parents_list[0], i, parents_list[1], j


def read_evidence(model):
    model_path = 'evaluation/evidence/uai/{}.uai.evid'.format(model)
    with open(model_path, 'r') as f:
        num_evidence_vars = int(f.readline())
        arr = f.readline().split()
        evidence = {}

        assert(len(arr) % 2 == 0)

        for i in range(len(arr) // 2):
            index, value = int(arr[i * 2]), int(arr[i * 2 + 1])
            evidence[index] = value

        return num_evidence_vars, evidence


def read_marginal(model):
    model_path = 'evaluation/marginals/uai/{}.uai.true.mar'.format(model)
    if not os.path.isfile(model_path):
        model_path = 'evaluation/marginals/uai/{}.uai.approx.mar'.format(model)
    with open(model_path, 'r') as f:
        result_type = f.readline()
        assert(result_type == 'MAR\n')

        lines = []
        for line in f:
            lines.append(line)

        marginal_line = None
        for line in reversed(lines):
            if line != '\n':
                marginal_line = line
                break

        assert(marginal_line is not None)

        arr = marginal_line.split()
        n = int(arr[0])
        marginals = {}

        var_index = -1
        for i in range(1, len(arr)):
            if i % 3 == 0:
                var_index += 1
                marginals[var_index] = float(arr[i])

        assert(len(marginals) == n)
        return marginals


def dump_network(model, n, parents, probs, num_evidence_vars, evidence, marginals, edges, nodes):
    parents = defaultdict(list, parents)
    data = {
        'n': n,
        'parents': parents,
        'probs': probs,
        'num_evidence_vars': num_evidence_vars,
        'evidence': evidence,
        'marginals': marginals,
        'edges': edges,
        'nodes': nodes
    }

    pickle.dump(data, open("evaluation/pickle/{}.pkl".format(model), "wb"))


def build_graph(n, parents):
    edges = defaultdict(list)
    for child, ancestors in parents.items():
        for parent in ancestors:
            edges[parent].append(child)

    nodes = topological_sort([i for i in range(n)], edges)

    return edges, nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', type=str)

    args = parser.parse_args()

    for model in args.models:
        n, parents, probs = read_model(model)
        num_evidence_vars, evidence = read_evidence(model)
        marginals = read_marginal(model)
        edges, nodes = build_graph(n, parents)
        dump_network(model, n, parents, probs, num_evidence_vars, evidence, marginals, edges, nodes)


if __name__ == "__main__":
    main()





from utils.directional_acyclic_graph_utils import DirectionalAcyclicGraphUtils


def main():
    adj = {0: [1, 2], 1: [3], 2: [3, 4], 3: [4], 4: []}

    labels = {
        0: "0 base input",
        1: "1 equidistant node generator",
        2: "2 function expression input",
        3: "3 interpolation values evaluator",
        4: "4 newton interpolation",
    }

    ascii_graph = DirectionalAcyclicGraphUtils.ascii_dag(adj, labels)
    print(ascii_graph)


if __name__ == "__main__":
    main()

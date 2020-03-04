import numpy as np


class Node:

    def __init__(self, name):
        self.name = name
        self.incoming = []
        self.outgoing = []
        # Own clique of given node, all that its connected to
        self.clique = []
        # probabilities = None

    def __str__(self):
        return "|" + self.name + "|"

    def print_clique(self):
        if self.clique == []:
            print(self)
        else:
            for inc_node in self.incoming:
                print(f"{inc_node} -> {self}")

            for out_node in self.outgoing:
                print(f"{self} -> {out_node}")

    def add_incoming(self, node):
        """
        adds incoming node
        :return:
        """
        self.incoming.append(node)
        self.clique.append(node)

    def add_outgoing(self, node):
        """
        adds outgoing node
        :return:
        """
        self.outgoing.append(node)
        self.clique.append(node)


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes

    def add_connection(self, node1: Node, node2: Node):
        # adds connection from node1 -> node2
        node1.add_outgoing(node2)
        node2.add_incoming(node1)

    def print_cliques(self):
        print("Printing Cliques:")
        covered_nodes = []
        for node in self.nodes:
            if node not in covered_nodes:
                node.print_clique()
                covered_nodes += node.clique



nodeA = Node("A")
nodeB = Node("B")
nodeC = Node("C")
nodes = [nodeA, nodeB, nodeC]
network = Graph(nodes)
network.print_cliques()
network.add_connection(nodeA, nodeB)
network.add_connection(nodeA, nodeC)
network.print_cliques()

print("Finished")

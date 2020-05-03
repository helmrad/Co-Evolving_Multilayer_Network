# -*- coding: utf-8 -*-
# Created with Python 3.6
"""
This code generates a co-evolving network.
An exchange of a quantity takes place on the network, while the weights of the links change randomly.
"""

# TODO
# Only monodirectional edges
# Insert markers between inter-layer nodes


import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
import numpy as np
from numpy import matlib as ml
import networkx as nx
import names
import copy
import itertools
import random


class CoEvolvingMultilayerNetwork:

    def __init__(self, layers, nodes):

        # Network parameters
        self.layers = layers
        self.nodes = nodes
        self.link_p = 0.5
        self.link_w_base = 0.25
        self.link_w_dev = 0.05
        self.node_q_init = 2

        # Drawing parameters
        self.node_size_min = 0.4
        self.node_fontsize = int(20*NODE_SIZE_MIN)
        self.node_size_scaling = 3000
        self.link_size_scaling = 10
        self.nodesets = layers + 1
        self.layer_coords = [[0, layer] for layer in range(0, LAYERS)]
        self.layer_coords.append([1, LAYERS-1])
        self.dynamic_node_range = range(LAYERS*NODES, (LAYERS+1)*NODES)
        self.colors = ['#003071', 'white']
        self.arrow = ArrowStyle('simple', head_length=25, head_width=20, tail_width=.75)

        # Simulation parameters
        self.iterations = 1000
        self.link_mod_base = 1
        self.link_mod_dev = 0.15
        self.node_q_min = 0.01
        self.node_q_max = 3
        self.link_w_min = 0.01
        self.link_w_max = 1

        # Optional node dynamics
        self.funcs = []
        func1 = lambda x: np.tanh(x)
        func2 = lambda x: -1*np.tanh(x)
        for n in range(self.nodes):
            self.funcs.append(func1) if np.random.uniform() > 0.5 else self.funcs.append(func2)

    def initialize_links(self):

        # Get adjacency tensor
        links = np.zeros((self.nodesets, self.nodes, self.nodes))
        for layer in range(0, self.layers):
            layer_links = np.random.uniform(self.link_w_base-self.link_w_dev, self.link_w_base+self.link_w_dev,
                                            size=(self.nodes, self.nodes))
            for i in range(self.nodes):
                for j in range(self.nodes):
                    if i == j or np.random.uniform() > self.link_p:
                        layer_links[i, j] = 0
            links[layer, :, :] = layer_links
        return links

    def initialize_labels_and_quantities(self):

        # Get names & quantities of nodes
        name_list, quantities = [], []
        for i in range(self.nodes):
            name = names.get_first_name()
            while name in name_list:
                name = names.get_first_name()
            name_list.append(name)
            quantities.append(np.random.uniform(0, self.node_q_init))
        name_list = self.layers*self.nodes * [' '] + name_list
        labels = dict(zip(range(0, self.nodesets*self.nodes), name_list))
        return labels, quantities

    def initialize_layout(self):
        # Get layouting parameters
        auxiliary_layers = [(nx.Graph(), self.layer_coords[i]) for i in range(0, self.nodesets)]
        # Nodes per layer
        ranges = [(self.nodes*i, self.nodes*(i+1)) for i in range(0, self.nodesets)]
        # Fill the auxiliary layers with nodes from the network
        for (aux_layer, coord), (start, end) in zip(auxiliary_layers, ranges):
            aux_layer.add_nodes_from(itertools.islice(self.net.nodes, start, end))
        # Calculate and move the nodes position
        all_pos = {}
        for aux_layer, (sx, sy) in auxiliary_layers:
            pos = nx.shell_layout(aux_layer, scale=2)  # or spring_layout...
            for node in pos:
                all_pos[node] = pos[node]
                all_pos[node] += (10 * sx, 10 * sy)
        return all_pos

    def initialize_colormap(self):
        # Generate colormap for node-coloring
        colors = []
        colors.append(["#{:06x}".format(random.randint(0, 0xFFFFFF)) for i in range(0, self.nodes)])
        colormap = self.nodesets * colors[0]
        return colormap

    def links_tensor_to_matrix(self):
        # Convert adjacency tensor to a matrix form, for networkx compliance
        links_mat = np.zeros((self.nodesets*self.nodes, self.nodesets*self.nodes))
        for n in range(0, self.layers):
            links_mat[self.nodes*n:self.nodes*(n+1), self.nodes*n:self.nodes*(n+1)] = self.links[2]
        return links_mat

    def links_matrix_to_array(self):
        # Flatten links matrix
        links_flat = []
        for lin in range(len(self.links)):
            for col in range(len(self.links)):
                if self.links[lin, col] != 0:
                    links_flat.append(self.links[lin, col])
        return links_flat

    def initialize_network(self):

        self.links = self.initialize_links()
        links_mat = self.links_tensor_to_matrix()
        self.net = nx.from_numpy_matrix(links_mat, create_using=nx.DiGraph())
        self.all_pos = self.initialize_layout()
        self.colormap = self.initialize_colormap()
        self.labels, self.quantities = self.initialize_labels_and_quantities()
        self.net = assign_quantities(self.net, self.quantities)

    def update_visualization(self):

        nodesizes = copy.deepcopy(self.quantities)
        for n in range(len(nodesizes)):
            nodesizes[n] = self.node_size_min if nodesizes[n] < self.node_size_min else nodesizes[n]
            nodesizes[n] = nodesizes[n]*self.node_size_scaling
        nodesizes = self.layers*self.nodes * [self.node_size_min*self.node_size_scaling] + nodesizes
        edges = self.net.edges()
        linkwidths = [self.net[u][v]['weight']*self.link_size_scaling for u, v in edges]

        # Update network visualization
        nx.draw(self.net, pos=self.all_pos, node_size=nodesizes, node_color=self.colormap,
                edges=edges, width=linkwidths, edge_color=self.colors[0],
                arrowstyle=self.arrow, arrowsize=.5, connectionstyle='arc3,rad=0.2',
                with_labels=False, font_color=self.colors[1], font_size=self.node_fontsize)
        nx.draw_networkx_labels(self.net, pos=self.all_pos, labels=self.labels,
                                font_color=self.colors[1], font_size=self.node_fontsize)

        txt = 'Total Quantity ' + str(np.round(np.sum(self.quantities), 2))
        #plt.text(1, 1, txt, horizontalalignment='center', verticalalignment='center', fontsize=20)
        #pylab.draw()
        plt.pause(.001)
        plt.clf()

    def network_dynamics(self):

        # Quantities
        # Amount of quantity transferred to each node
        # transfers = np.dot(quantities, links)
        # # Amount of quantity lost at each node
        # quantities_rep = ml.repmat(quantities, len(quantities), 1)
        # losses = np.diagonal(np.dot(links, quantities_rep))
        # # Merge losses & transfers
        # #quantities = [quantities[n] - losses[n] + funcs[n](transfers[n]) for n in range(len(name_list))]
        # quantities = [quantities[n] - losses[n] + transfers[n] for n in range(NODESETS*NODES)]
        self.quantities = [quantity + np.random.uniform(-1, 1)*0.05 for quantity in self.quantities]
        # Links: restore & alter
        self.links = np.array(nx.convert_matrix.to_numpy_matrix(self.net))
        self.links = np.multiply(self.links, np.random.uniform(self.link_mod_base-self.link_mod_dev,
                                                               self.link_mod_base+self.link_mod_dev,
                                                               size=self.links.shape))

    def assign_quantities(self):
        quantities_dict = dict(zip(self.dynamic_node_range, self.quantities))
        nx.set_node_attributes(self.net, quantities_dict, 'Quantities')

    def limit_params(self, links_flat):
        # Constraining both the quantity and the link weights
        for params, boundaries in zip([self.quantities, links_flat],
                                      [[self.node_q_min, self.node_q_max], [self.link_w_min, self.link_w_max]]):
            for p in range(len(params)):
                if boundaries[0] < params[p] < boundaries[1]:
                    params[p] = params[p]
                elif params[p] < boundaries[0]:
                    params[p] = boundaries[0]
                elif params[p] > boundaries[1]:
                    params[p] = boundaries[1]
        return links_flat

    def constrain_and_reassign(self):

        links_flat = self.links_matrix_to_array()
        links_flat = self.limit_params(links_flat)

        assign_quantities(self.net, self.quantities)
        # Reassign weights to links
        for edge_triple, link in zip(self.net.edges(data=True), links_flat):
            edge_triple[-1]['weight'] = link

    def simulate_network(self):

        pylab.ion()
        fig = plt.figure(0, figsize=(16, 8))
        fig.canvas.set_window_title('Propagation of a Quantity on a Dynamic Network')
        self.initialize_network()

        # Repeatedly run dynamics and update visualization
        for i in range(self.iterations):

            self.update_visualization()
            # Get quantities
            # quantities_dict = nx.get_node_attributes(self.net, 'Quantities')
            # quantities = [quantities_dict[node] for node in DYNAMIC_NODE_RANGE]
            # update_visualization(net, all_pos, quantities, labels, colormap)

            # Simulate dynamics
            # quantities, links = network_dynamics(quantities, links)
            self.network_dynamics()
            # Constrain both quantities and links to a certain level, and reassign to the network
            self.constrain_and_reassign()

            # If user stops visualization, terminate the script
            if not plt.fignum_exists(0):
                break

# Declare network parameters
LAYERS = 3
NODES = 6
LINK_P = 0.5
LINK_W_BASE = 0.25
LINK_W_DEV = 0.05
NODE_Q_INIT = 2

# Drawing parameters
NODE_SIZE_MIN = 0.4
NODE_FONT_SIZE = int(20*NODE_SIZE_MIN)
NODE_SIZE_SCALING = 3000
LINK_SIZE_SCALING = 10
NODESETS = LAYERS + 1
LAYER_COORDS = [[0, layer] for layer in range(0, LAYERS)]
LAYER_COORDS.append([1, LAYERS-1])
DYNAMIC_NODE_RANGE = range(LAYERS*NODES, (LAYERS+1)*NODES)

# Simulation parameters
ITERATIONS = 1000
LINK_MOD_BASE = 1
LINK_MOD_DEV = 0.15
NODE_Q_MIN = 0.01
NODE_Q_MAX = 3
LINK_W_MIN = 0.01
LINK_W_MAX = 1

# Optional node dynamics
funcs = []
func1 = lambda x: np.tanh(x)
func2 = lambda x: -1*np.tanh(x)
for n in range(NODES):
    funcs.append(func1) if np.random.uniform() > 0.5 else funcs.append(func2)


def initialize_network():

    # Get names & quantities of nodes
    name_list, quantities = [], []
    for i in range(NODES):
        name = names.get_first_name()
        while name in name_list:
            name = names.get_first_name()
        name_list.append(name)
        quantities.append(np.random.uniform(0, NODE_Q_INIT))
    name_list = LAYERS*NODES * [' '] + name_list
    labels = dict(zip(range(0, NODESETS*NODES), name_list))

    # Get adjacency tensor
    links = np.zeros((NODESETS, NODES, NODES))
    for layer in range(0, LAYERS):
        layer_links = np.random.uniform(LINK_W_BASE-LINK_W_DEV, LINK_W_BASE+LINK_W_DEV, size=(NODES, NODES))
        for i in range(NODES):
            for j in range(NODES):
                if i == j or np.random.uniform() > LINK_P:
                    layer_links[i,j] = 0
        links[layer, :, :] = layer_links

    # Build network from adjacency matrix
    links_mat = tensor_to_matrix(links)
    net = nx.from_numpy_matrix(links_mat, create_using=nx.DiGraph())
    # Assign quantities
    net = assign_quantities(net, quantities)

    # Get layouting parameters
    auxiliary_layers = [(nx.Graph(), LAYER_COORDS[i]) for i in range(0, NODESETS)]
    # Nodes per layer
    ranges = [(NODES*i, NODES*(i+1)) for i in range(0, NODESETS)]
    # Fill the auxiliary layers with nodes from the network
    for (aux_layer, coord), (start, end) in zip(auxiliary_layers, ranges):
        aux_layer.add_nodes_from(itertools.islice(net.nodes, start, end))
    # Calculate and move the nodes position
    all_pos = {}
    for aux_layer, (sx, sy) in auxiliary_layers:
        pos = nx.shell_layout(aux_layer, scale=2)  # or spring_layout...
        for node in pos:
            all_pos[node] = pos[node]
            all_pos[node] += (10 * sx, 10 * sy)

    # Generate colormap for node-coloring
    colors = []
    colors.append(["#{:06x}".format(random.randint(0, 0xFFFFFF)) for i in range(0, NODES)])
    colormap = NODESETS * colors[0]

    return net, all_pos, labels, colormap

def update_visualization(net, all_pos, quantities, labels, colormap):

    colors = ['black', '#003071', 'white']
    arrow = ArrowStyle('simple', head_length=25, head_width=20, tail_width=.75)

    nodesizes = copy.deepcopy(quantities)
    for n in range(len(quantities)):
        nodesizes[n] = NODE_SIZE_MIN if nodesizes[n] < NODE_SIZE_MIN else nodesizes[n]
        nodesizes[n] = nodesizes[n]*NODE_SIZE_SCALING
    nodesizes = LAYERS*NODES * [NODE_SIZE_MIN*NODE_SIZE_SCALING] + nodesizes
    edges = net.edges()
    linkwidths = [net[u][v]['weight']*LINK_SIZE_SCALING for u, v in edges]

    # Update network visualization
    nx.draw(net, pos=all_pos, node_size=nodesizes, node_color=colormap,
            edges=edges, width=linkwidths, edge_color=colors[1],
            arrowstyle=arrow, arrowsize=.5, connectionstyle='arc3,rad=0.2',
            with_labels=False, font_color=colors[2], font_size=NODE_FONT_SIZE)
    nx.draw_networkx_labels(net, pos=all_pos, labels=labels,
                            font_color=colors[2], font_size=NODE_FONT_SIZE)

    txt = 'Total Quantity ' + str(np.round(np.sum(quantities), 2))
    #plt.text(1, 1, txt, horizontalalignment='center', verticalalignment='center', fontsize=20)
    #pylab.draw()
    plt.pause(.001)
    plt.clf()

def network_dynamics(quantities, links):

    # Quantities
    # Amount of quantity transferred to each node
    # transfers = np.dot(quantities, links)
    # # Amount of quantity lost at each node
    # quantities_rep = ml.repmat(quantities, len(quantities), 1)
    # losses = np.diagonal(np.dot(links, quantities_rep))
    # # Merge losses & transfers
    # #quantities = [quantities[n] - losses[n] + funcs[n](transfers[n]) for n in range(len(name_list))]
    # quantities = [quantities[n] - losses[n] + transfers[n] for n in range(NODESETS*NODES)]
    quantities = [quantity + np.random.uniform(-1, 1)*0.05 for quantity in quantities]
    # Links
    links = np.multiply(links, np.random.uniform(LINK_MOD_BASE-LINK_MOD_DEV, LINK_MOD_BASE+LINK_MOD_DEV,
                                                 size=(links.shape)))
    return quantities, links

def assign_quantities(net, quantities):
    quantities_dict = dict(zip(DYNAMIC_NODE_RANGE, quantities))
    nx.set_node_attributes(net, quantities_dict, 'Quantities')
    return net

def flatten(links):
    # Flatten links matrix
    links_flat = []
    for lin in range(len(links)):
        for col in range(len(links)):
            if links[lin, col] != 0:
                links_flat.append(links[lin, col])
    return links_flat

def limit_params(quantities, links_flat):
    # Constraining both the quantity and the link weights
    for params, boundaries in zip([quantities, links_flat], [[NODE_Q_MIN, NODE_Q_MAX], [LINK_W_MIN, LINK_W_MAX]]):
        for p in range(len(params)):
            if params[p] > boundaries[0] and params[p] < boundaries[1]:
                params[p] = params[p]
            elif params[p] < boundaries[0]:
                params[p] = boundaries[0]
            elif params[p] > boundaries[1]:
                params[p] = boundaries[1]
    return quantities, links_flat

def tensor_to_matrix(links):
    # Convert adjacency tensor to a matrix form, for networkx compliance
    links_mat = np.zeros((NODESETS*NODES, NODESETS*NODES))
    for n in range(0, LAYERS):
        links_mat[NODES*n:NODES*(n+1),NODES*n:NODES*(n+1)] = links[2]
    return links_mat


if __name__ == "__main__":

    network = CoEvolvingMultilayerNetwork(layers=3, nodes=6)
    network.simulate_network()
    # pylab.ion()
    # fig = plt.figure(0, figsize=(16, 8))
    # fig.canvas.set_window_title('Propagation of a Quantity on a Dynamic Network')
    # net, all_pos, labels, colormap = initialize_network()
    #
    # # Repeatedly run dynamics and update visualization
    # for i in range(ITERATIONS):
    #
    #     # Get quantities
    #     quantities_dict = nx.get_node_attributes(net, 'Quantities')
    #     quantities = [quantities_dict[node] for node in DYNAMIC_NODE_RANGE]
    #     update_visualization(net, all_pos, quantities, labels, colormap)
    #
    #     # Get links
    #     links = np.array(nx.convert_matrix.to_numpy_matrix(net))
    #     # Simulate dynamics
    #     quantities, links = network_dynamics(quantities, links)
    #     # Constrain both quantities and links to a certain level
    #     links_flat = flatten(links)
    #     quantities, links_flat = limit_params(quantities, links_flat)
    #
    #     net = assign_quantities(net, quantities)
    #     # Reassign weights to links
    #     for edge_triple, link in zip(net.edges(data=True), links_flat):
    #         edge_triple[-1]['weight'] = link
    #
    #     # If user stops visualization, terminate the script
    #     if not plt.fignum_exists(0):
    #         break

# -*- coding: utf-8 -*-
# Created with Python 3.6
"""
This code generates a co-evolving network.
An exchange of a quantity takes place on the network, while the weights of the links change randomly.
"""

# TODO
# implement multilayer dynamics
# try colored interlayer-links
# support different layouts
# make dynamic view larger

# TODO Project
# Assign names of banks (maybe automated like bank of <random European country>)
# Put a map of Europe below and use markers to affiliate nodes with countries
# Three layers as given by Thurner
# Graphs on the right, like total quantity over time and stock indices
# Dynamics

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


class CoevolvingMultilayerNetwork:

    def __init__(self, layers, nodes, link_p):
        # Network parameters
        self.layers = layers
        self.nodes = nodes
        self.link_p = link_p
        self.link_w_base = 0.25
        self.link_w_dev = 0.05
        self.node_q_init = 2

        # Drawing parameters
        self.node_size_min = 0.4
        self.node_fontsize = int(20*self.node_size_min)
        self.static_node_size = 0.15
        self.node_size_scaling = 3000
        self.link_size_scaling = 10
        self.nodesets = layers + 1
        self.layer_coords = [[0, layer] for layer in range(0, self.layers)]
        self.layer_coords.append([1, self.layers-1])
        self.dynamic_node_range = range(self.layers*self.nodes, (self.layers+1)*self.nodes)
        self.colors = ['#003071', 'white', 'grey']
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
                    if i == j or layer_links[j, i] != 0 or np.random.uniform() > self.link_p:
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
        colors = [["#{:06x}".format(random.randint(0, 0xFFFFFF)) for i in range(0, self.nodes)]]
        colormap = self.nodesets * colors[0]
        return colormap

    def initialize_interlayer_links(self):
        # Generate the links that indicate shared identity of nodes in different layers
        links_interlayer = np.zeros((self.nodesets*self.nodes, self.nodesets*self.nodes))
        for l in range(0, self.layers-1):
            for n in range(0, self.nodes):
                links_interlayer[l*self.nodes+n, (l+1)*self.nodes+n] = 1
        auxiliary_net = nx.from_numpy_matrix(links_interlayer, create_using=nx.Graph())
        links_interlayer = auxiliary_net.edges
        return links_interlayer

    def links_tensor_to_matrix(self):
        # Convert adjacency tensor to a matrix form, for networkx compliance
        links_mat = np.zeros((self.nodesets*self.nodes, self.nodesets*self.nodes))
        for n in range(0, self.layers):
            links_mat[self.nodes*n:self.nodes*(n+1), self.nodes*n:self.nodes*(n+1)] = self.links[n, :, :]
        return links_mat

    def links_matrix_to_tensor(self, links_mat):
        links = np.zeros((self.nodesets, self.nodes, self.nodes))
        for n in range(0, self.layers):
            links[n, :, :] = links_mat[self.nodes*n:self.nodes*(n+1), self.nodes*n:self.nodes*(n+1)]
        return links

    def links_matrix_to_array(self):
        # Flatten links matrix
        links_flat = []
        for lin in range(len(self.links)):
            for col in range(len(self.links)):
                if self.links[lin, col] != 0:
                    links_flat.append(self.links[lin, col])
        return links_flat

    def initialize(self):
        self.links = self.initialize_links()
        links_mat = self.links_tensor_to_matrix()
        # links = self.links_matrix_to_tensor(links_mat)
        self.net = nx.from_numpy_matrix(links_mat, create_using=nx.DiGraph())
        self.all_pos = self.initialize_layout()
        self.colormap = self.initialize_colormap()
        self.labels, self.quantities = self.initialize_labels_and_quantities()
        self.links_interlayer = self.initialize_interlayer_links()
        self.assign_quantities()

    def update_visualization(self):
        nodesizes = copy.deepcopy(self.quantities)
        for n in range(len(nodesizes)):
            nodesizes[n] = self.node_size_min if nodesizes[n] < self.node_size_min else nodesizes[n]
            nodesizes[n] = nodesizes[n]*self.node_size_scaling
        nodesizes = self.layers*self.nodes * [self.static_node_size*self.node_size_scaling] + nodesizes
        edges = self.net.edges()
        linkwidths = [self.net[u][v]['weight']*self.link_size_scaling for u, v in edges]

        # Update network visualization
        nx.draw(self.net, pos=self.all_pos, node_size=nodesizes, node_color=self.colormap,
                edges=edges, width=linkwidths, edge_color=self.colors[0],
                arrowstyle=self.arrow, arrowsize=.5, connectionstyle='arc3,rad=0.2',
                with_labels=False, font_color=self.colors[1], font_size=self.node_fontsize)
        nx.draw_networkx_labels(self.net, pos=self.all_pos, labels=self.labels,
                                font_color=self.colors[1], font_size=self.node_fontsize)
        nx.draw_networkx_edges(self.net, pos=self.all_pos, edgelist=self.links_interlayer,
                               style='dotted', edge_color=self.colors[2], alpha=.5, arrows=False)

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
        # Reassign quantities to nodes and weights to links
        self.assign_quantities()
        for edge_triple, link in zip(self.net.edges(data=True), links_flat):
            edge_triple[-1]['weight'] = link

    def simulate(self):
        pylab.ion()
        fig = plt.figure(0, figsize=(16, 8))
        fig.canvas.set_window_title('Propagation of a Quantity on a Dynamic Network')
        # Repeatedly run dynamics and update visualization
        for i in range(self.iterations):
            self.update_visualization()
            self.network_dynamics()
            self.constrain_and_reassign()
            # If user stops visualization, terminate the script
            if not plt.fignum_exists(0):
                break


if __name__ == "__main__":
    network = CoevolvingMultilayerNetwork(layers=3, nodes=20, link_p=.15)
    network.initialize()
    network.simulate()

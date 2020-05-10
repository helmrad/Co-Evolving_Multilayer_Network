# -*- coding: utf-8 -*-
# Created with Python 3.6
"""
This code generates a co-evolving network.
An exchange of a quantity takes place on the network, while the weights of the links change randomly.
"""

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
import networkx as nx
import names
import copy
import itertools
import random
import matplotlib.image as mpimg


class CoevolvingMultilayerNetwork:

    def __init__(self, layers, nodes, link_p):
        # Initialize parameters
        # Network parameters
        self.layers = layers
        self.nodes = nodes
        self.link_p = link_p
        self.link_w_base = 0.25
        self.link_w_dev = 0.05
        self.node_q_init = 2

        # Drawing parameters
        self.node_size_min = 0.4
        self.node_font_sizes = int(20*self.node_size_min)
        self.mln_node_size = 0.15
        self.node_size_scaling = 3000
        self.link_size_scaling = 10
        self.mln_node_sizes = self.layers*self.nodes * [self.mln_node_size*self.node_size_scaling]
        self.dynamic_node_range = range(self.layers*self.nodes, (self.layers+1)*self.nodes)
        self.colors = ['#003071', 'white', 'grey']
        self.arrow = ArrowStyle('simple', head_length=25, head_width=20, tail_width=.75)
        self.layer_y_dist = 15
        self.layer_scale = 3
        self.dynamic_node_scale = 3
        self.dynamic_node_lim_x_offset = 1
        self.dynamic_node_lim_y_offset = 3
        self.txt_font_size = 15
        self.mln_idx = self.nodes*self.layers

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

        # Set up the figure
        pylab.ion()
        self.fig = plt.figure(0, figsize=(16, 8))
        self.fig.canvas.set_window_title('Co-evolving Multilayer Network')
        self.ax_mln = plt.subplot2grid((5, 2), (0, 0), colspan=1, rowspan=5)
        self.ax_nodes = plt.subplot2grid((5, 2), (0, 1), colspan=1, rowspan=2)
        self.ax_3 = plt.subplot2grid((5, 2), (2, 1), colspan=1, rowspan=3)
        self.fig.tight_layout(pad=3, h_pad=1.25, w_pad=1.25)

    def initialize_links(self):
        # Compute an adjacency tensor for multilayer connectivity
        links = np.zeros((self.layers, self.nodes, self.nodes))
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
        # Get random labels & quantities of nodes
        name_list, quantities = [], []
        for i in range(self.nodes):
            name = names.get_first_name()
            while name in name_list:
                name = names.get_first_name()
            name_list.append(name)
            quantities.append(np.random.uniform(0, self.node_q_init))
        labels = dict(zip(range(0, self.nodes), name_list))
        return labels, quantities

    def initialize_layout(self):
        # Calculate the positions of all nodes in the network visualization
        # Generate an auxiliary graph and fill it with nodes from the network
        auxiliary_graph = nx.Graph()
        auxiliary_graph.add_nodes_from(itertools.islice(self.net.nodes, 0, self.nodes))
        # Set the layout for one layer, and offset the positions of nodes in each other layer accordingly
        all_pos = {}
        pos = nx.shell_layout(auxiliary_graph)
        for l in range(0, self.layers):
            for node in pos:
                all_pos[l*self.nodes+node] = pos[node]*self.layer_scale + (0, l*self.layer_y_dist)
        return all_pos

    def arrange_subplots(self):
        # Tune the limits and scales in the subplots
        # Tune the axis on which the multilayer network is displayed
        mln_y_min, mln_y_max = np.inf, -np.inf
        for n in range(0, self.layers * self.nodes):
            _, y = self.all_pos[n]
            mln_y_min = y if y < mln_y_min else mln_y_min
            mln_y_max = y if y > mln_y_max else mln_y_max
        mln_ylims = (mln_y_min - (mln_y_max - mln_y_min)/(2*self.layers),
                     mln_y_max + (mln_y_max - mln_y_min)/(2*self.layers))
        self.ax_mln.set_ylim(mln_ylims)

        # Tune the axis that shows the evolution of the nodes
        dynamic_node_x_min, dynamic_node_x_max, dynamic_node_y_min, dynamic_node_y_max = np.inf, -np.inf, np.inf, -np.inf
        for n in range(0, self.nodes):
            x, y = self.all_pos[n]
            dynamic_node_x_min = x if x < dynamic_node_x_min else dynamic_node_x_min
            dynamic_node_x_max = x if x > dynamic_node_x_max else dynamic_node_x_max
            dynamic_node_y_min = y if y < dynamic_node_y_min else dynamic_node_y_min
            dynamic_node_y_max = y if y > dynamic_node_y_max else dynamic_node_y_max
        dn_xlims = (dynamic_node_x_min - self.dynamic_node_lim_x_offset,
                    dynamic_node_x_max + self.dynamic_node_lim_x_offset)
        dn_ylims = (dynamic_node_y_min - self.dynamic_node_lim_y_offset,
                    dynamic_node_y_max + self.dynamic_node_lim_y_offset)
        self.ax_nodes.set_xlim(dn_xlims)
        self.ax_nodes.set_ylim(dn_ylims)
        # Remove boxes around axes
        self.ax_nodes.axis('off')
        self.ax_3.axis('off')

    def initialize_colormap(self):
        # Generate colormap for node-coloring
        colors = [["#{:06x}".format(random.randint(0, 0xFFFFFF)) for i in range(0, self.nodes)]]
        colormap = self.layers * colors[0]
        return colormap

    def initialize_interlayer_links(self):
        # Generate the links that indicate shared identity of nodes in different layers
        links_interlayer = np.zeros((self.layers*self.nodes, self.layers*self.nodes))
        for l in range(0, self.layers-1):
            for n in range(0, self.nodes):
                links_interlayer[l*self.nodes+n, (l+1)*self.nodes+n] = 1
        auxiliary_net = nx.from_numpy_matrix(links_interlayer, create_using=nx.Graph())
        links_interlayer = auxiliary_net.edges
        return links_interlayer

    def links_tensor_to_matrix(self):
        # Convert adjacency tensor to a matrix form, for networkx compliance
        links_mat = np.zeros((self.layers*self.nodes, self.layers*self.nodes))
        for n in range(0, self.layers):
            links_mat[self.nodes*n:self.nodes*(n+1), self.nodes*n:self.nodes*(n+1)] = self.links[n, :, :]
        return links_mat

    def links_matrix_to_tensor(self, links_mat):
        # Convert networkx-compliant matrix form to an adjancecy tensor
        links = np.zeros((self.layers, self.nodes, self.nodes))
        for n in range(0, self.layers):
            links[n, :, :] = links_mat[self.nodes*n:self.nodes*(n+1), self.nodes*n:self.nodes*(n+1)]
        return links

    def links_matrix_to_array(self, links_mat):
        # Convert a matrix to a 1d list
        links_flat = []
        for lin in range(len(links_mat)):
            for col in range(len(links_mat)):
                if links_mat[lin, col] != 0:
                    links_flat.append(links_mat[lin, col])
        return links_flat

    def draw_labels(self):
        # Draw labels that correspond to nodes
        nx.draw_networkx_labels(self.net, pos=self.all_pos, labels=self.labels, ax=self.ax_nodes,
                                font_color=self.colors[1], font_size=self.node_font_sizes, font_family='verdana')

    def initialize(self):
        # A second initialization (next to __init__) for all parameters and variables that are more complex
        self.links = self.initialize_links()
        links_mat = self.links_tensor_to_matrix()
        self.net = nx.from_numpy_matrix(links_mat, create_using=nx.DiGraph())
        self.dynamic_nodes = nx.from_numpy_matrix(np.zeros((self.nodes, self.nodes)))
        self.all_pos = self.initialize_layout()
        self.arrange_subplots()
        self.colormap = self.initialize_colormap()
        self.labels, self.quantities = self.initialize_labels_and_quantities()
        self.draw_labels()
        self.links_interlayer = self.initialize_interlayer_links()
        self.assign_quantities()

    def update_visualization(self):
        # Calculate sizes of links and nodes, and update the visualization of the network
        edges = self.net.edges()
        linkwidths = [self.net[u][v]['weight']*self.link_size_scaling for u, v in edges]
        node_sizes = copy.deepcopy(self.quantities)
        for n in range(len(node_sizes)):
            node_sizes[n] = self.node_size_min if node_sizes[n] < self.node_size_min else node_sizes[n]
            node_sizes[n] = node_sizes[n]*self.node_size_scaling

        # Update network visualization
        # Draw multilayer network
        nx.draw(self.net, pos=self.all_pos, node_size=self.mln_node_sizes, node_color=self.colormap,
                edges=edges, width=linkwidths, edge_color=self.colors[0], ax=self.ax_mln,
                arrowstyle=self.arrow, arrowsize=.5, connectionstyle='arc3,rad=0.2', with_labels=False)
        # Draw markers between identical nodes in multilayer network
        nx.draw_networkx_edges(self.net, pos=self.all_pos, edgelist=self.links_interlayer, ax=self.ax_mln,
                               style='dotted', edge_color=self.colors[2], alpha=.5, arrows=False)
        # Draw dynamic nodes
        nx.draw_networkx_nodes(self.dynamic_nodes, pos=self.all_pos, node_size=node_sizes,
                               node_color=self.colormap[:self.nodes], ax=self.ax_nodes, with_labels=False)

        # Animate the entire amount of quantity in the network
        txt = 'Entire quantity ' + str(np.round(np.sum(self.quantities), 2))
        txtvar = self.ax_3.text(0.5, 0.5, txt, horizontalalignment='center',
                           verticalalignment='center', fontsize=self.txt_font_size)
        plt.pause(.001)
        # Clear everything besides the image
        for artist in self.fig.axes[0].collections + self.fig.axes[0].patches + self.fig.axes[1].collections:
            artist.remove()
        txtvar.set_visible(False)

    def network_dynamics(self):
        # Co-evolving dynamics
        # Restore links in tensor form
        self.links = self.links_matrix_to_tensor(nx.convert_matrix.to_numpy_matrix(self.net))
        # Random quantitiy dynamics
        self.quantities = [quantity + np.random.uniform(-1, 1)*0.05 for quantity in self.quantities]
        # Random link dynamics
        self.links = np.multiply(self.links, np.random.uniform(self.link_mod_base-self.link_mod_dev,
                                                               self.link_mod_base+self.link_mod_dev,
                                                               size=self.links.shape))

    def assign_quantities(self):
        # Assign quantities to nodes
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
        # Constrain both links and quantities, and reassign to the network
        links_flat = self.links_matrix_to_array(self.links_tensor_to_matrix())
        links_flat = self.limit_params(links_flat)
        # Reassign quantities to nodes and weights to links
        self.assign_quantities()
        for edge_triple, link in zip(self.net.edges(data=True), links_flat):
            edge_triple[-1]['weight'] = link

    def simulate(self):
        # Start simulating the dynamics and updating the visualization
        # Repeatedly run dynamics and update visualization
        for i in range(self.iterations):
            self.update_visualization()
            self.network_dynamics()
            self.constrain_and_reassign()
            # If user stops visualization, terminate the script
            if not plt.fignum_exists(0):
                break


if __name__ == "__main__":
    # Set up & simulate the co-evolving multilayer network
    network = CoevolvingMultilayerNetwork(layers=3, nodes=15, link_p=.15)
    network.initialize()
    network.simulate()

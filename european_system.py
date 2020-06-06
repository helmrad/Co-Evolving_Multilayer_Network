# -*- coding: utf-8 -*-
# Created with Python 3.7
"""
This code generates a simulation of European economic interactions.
A quantity is exchanged, consumed, and produced by the individual countries.
"""


import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle, Polygon
import numpy as np
import networkx as nx
import copy
import random
import pandas as pd
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class EuropeanSystem:

    def __init__(self, layers, nodes, link_p):
        # Initialize parameters
        # Network parameters
        self.layers = layers
        self.nodes = nodes
        self.link_p = link_p
        self.link_w_base = 0.25
        self.link_w_dev = 0.05
        self.node_q_init = 2

        # Simulation parameters
        self.iterations = 500
        self.link_mod_base = 1
        self.link_mod_dev = 0.15
        self.node_q_min = 0.01
        self.node_q_max = 3
        self.link_w_min = 0.01
        self.link_w_max = 1

        # Drawing parameters
        self.node_size_min = 0.4
        self.node_font_sizes = int(25*self.node_size_min)
        self.mln_node_size = 0.3
        self.node_size_scaling = 1000
        self.link_size_scaling = 3
        self.arrow = ArrowStyle('simple', head_length=10, head_width=10, tail_width=.75)
        self.mln_node_sizes = self.layers*self.nodes * [self.mln_node_size*self.node_size_scaling]
        self.dynamic_node_range = range(self.layers*self.nodes, (self.layers+1)*self.nodes)
        self.colors = ['#003071', 'white', 'grey', 'black']
        self.layer_y_dist = 11.5
        self.map_layer_dist = 15
        self.mln_xlims = (-4, 4)
        self.mln_ylims = (-20, 30)
        self.img_size_orig = (1160, 644)
        self.img_size = (850, 335)
        self.img_zoom = 0.72
        self.img_xy = [-0.5, -14]
        self.pll_x = [-3.1, 2.8, 3.3, -2.6]
        self.pll_y = [-4.4, -4.4, 6.6, 6.6]
        self.dynamic_node_scale = 3
        self.dynamic_node_lim_x_offset = 1
        self.dynamic_node_lim_y_offset = 3
        self.ax_3_xlims = (0, self.iterations)
        self.ax_3_ylims = (0, self.node_q_init*self.nodes)
        self.tick_size = 13
        self.label_font_size = 17
        self.ax_3_lw = 1

        # Set up the figure
        pylab.ion()
        self.dpi = 100
        self.fig_size = (19.2, 10.8)
        self.resolution = [f * self.dpi for f in self.fig_size]
        self.fig = plt.figure(0, figsize=self.fig_size, dpi=self.dpi)
        self.fig.canvas.set_window_title('European System')
        self.ax_mln = plt.subplot2grid((5, 10), (0, 0), rowspan=5, colspan=5)
        self.ax_nodes = plt.subplot2grid((5, 10), (0, 5), rowspan=2, colspan=5)
        self.ax_3 = plt.subplot2grid((5, 10), (2, 6), rowspan=2, colspan=3)
        self.fig.tight_layout(pad=3, h_pad=1.25, w_pad=1.25)

    def initialize_links(self):
        # Compute an adjacency tensor for multilayer connectivity
        links = np.zeros((self.layers, self.nodes, self.nodes))
        for layer in range(0, self.layers):
            layer_links = np.random.uniform(self.link_w_base-self.link_w_dev, self.link_w_base+self.link_w_dev,
                                            size=(self.nodes, self.nodes))
            for i in range(self.nodes):
                for j in range(self.nodes):
                    if i == j or np.random.uniform() > self.link_p:
                        layer_links[i, j] = 0
            links[layer, :, :] = layer_links
        return links

    def initialize_quantities(self):
        quantities = []
        for i in range(self.nodes):
            quantities.append(np.random.uniform(0, self.node_q_init))
        return quantities

    def get_layouting(self):
        # Read in data related to the map
        data = pd.read_excel('data/countries.xlsx', nrows=self.nodes)
        name_list = list(data['Code'].values)
        labels = dict(zip(range(0, self.nodes), name_list))
        colors = list(data['Color'].values)
        px, py = data['px'].values, data['py'].values
        # Generate colormap for node-coloring
        color_list = self.layers * colors
        # Calculate the positions of all nodes in the network visualization
        all_pos = {}
        for l in range(0, self.layers):
            for node in range(0, self.nodes):
                all_pos[l*self.nodes+node] = (px[node], py[node] + self.map_layer_dist + l*self.layer_y_dist)
        aux_all_pos = dict(zip(range(0, self.nodes), [(x, y) for x, y in zip(px, py)]))
        # Get positions for an auxiliary network that is used to visualize the interlayer links
        for l in range(1, self.layers+1):
            for node in range(0, self.nodes):
                aux_all_pos[l*self.nodes+node] = (px[node], py[node] + self.map_layer_dist + l*self.layer_y_dist)
        return labels, color_list, all_pos, aux_all_pos

    def arrange_subplots(self):
        # Tune the limits and scales in the subplots
        # Tune the axis on which the multilayer network is displayed
        self.ax_mln.set_xlim(self.mln_xlims)
        self.ax_mln.set_ylim(self.mln_ylims)

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

        # Tune the axis that plots the time series
        self.ax_3.tick_params(axis='both', which='both', labelsize=self.tick_size)
        self.ax_3.set_xlabel('Time [days]', fontsize=self.label_font_size)
        self.ax_3.set_ylabel('Aggregate quantity', fontsize=self.label_font_size)
        self.ax_3.set_xlim(self.ax_3_xlims)
        self.ax_3.set_ylim(self.ax_3_ylims)

        # Remove boxes around axes
        self.ax_mln.axis('off')
        self.ax_nodes.axis('off')

    def insert_map(self):
        # Insert the image of the map
        # Read image
        file = 'images/europe_map.png'
        img = Image.open(file)
        img = img.resize(self.img_size)
        arr_img = np.asarray(img)
        # Insert image
        imagebox = OffsetImage(arr_img, zoom=self.img_zoom, zorder=0)
        imagebox.image.axes = self.ax_mln
        ab = AnnotationBbox(imagebox, self.img_xy, frameon=False)
        self.ax_mln.add_artist(ab)
        # Set the map to the background
        self.ax_mln.artists[0].zorder = 0

    def initialize_colormap(self):
        # Generate colormap for node-coloring
        colors = [["#{:06x}".format(random.randint(0, 0xFFFFFF)) for i in range(0, self.nodes)]]
        colormap = self.layers * colors[0]
        return colormap

    def initialize_interlayer_links(self):
        # Generate the links that indicate shared identity of nodes in different layers
        links_interlayer = np.zeros(((self.layers+1)*self.nodes, (self.layers+1)*self.nodes))
        for l in range(0, self.layers-1):
            for n in range(0, self.nodes):
                links_interlayer[l*self.nodes+n, (l+1)*self.nodes+n] = 1
        # Generate an auxiliary network for networkx compliance
        auxiliary_net = nx.from_numpy_matrix(links_interlayer, create_using=nx.Graph())
        links_interlayer = auxiliary_net.edges
        return auxiliary_net, links_interlayer

    def initialize_time_series(self):
        # Initialize the line that is shown as time series
        time_series_x = np.arange(0, self.iterations)
        time_series_y = []
        line, = self.ax_3.plot(time_series_x, np.zeros(self.iterations), color=self.colors[3], linewidth=self.ax_3_lw)
        return time_series_x, time_series_y, line

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

    def initialize_mln(self):
        # Initialize layers, nodes and intra-layer links of the multilayer network
        # Draw parallelograms for separating the layers
        for l in range(self.layers):
            pll_y = [y + l*self.layer_y_dist for y in self.pll_y]
            self.ax_mln.add_patch(Polygon(xy=list(zip(self.pll_x, pll_y)), fill=True, color=self.colors[2], alpha=.1))
        # Draw nodes
        nx.draw_networkx_nodes(self.net, pos=self.all_pos, node_soze=self.mln_node_sizes, node_color=self.color_list,
                               ax=self.ax_mln, with_labels=False)
        self.ax_mln.collections[0].zorder = 3
        # Draw markers between identical nodes in multilayer network
        nx.draw_networkx_edges(self.auxiliary_net, pos=self.aux_all_pos, edgelist=self.links_interlayer,
                               ax=self.ax_mln, style='dotted', edge_color=self.colors[2], alpha=.75, arrows=False)
        self.ax_mln.collections[1].zorder = 2

    def initialize(self):
        # A second initialization (next to __init__) for all parameters and variables that are more complex
        self.links = self.initialize_links()
        links_mat = self.links_tensor_to_matrix()
        self.net = nx.from_numpy_matrix(links_mat, create_using=nx.DiGraph())
        self.dynamic_nodes = nx.from_numpy_matrix(np.zeros((self.nodes, self.nodes)))
        self.quantities = self.initialize_quantities()
        self.labels, self.color_list, self.all_pos, self.aux_all_pos = self.get_layouting()
        self.auxiliary_net, self.links_interlayer = self.initialize_interlayer_links()
        self.time_series_x, self.time_series_y, self.line = self.initialize_time_series()
        self.arrange_subplots()
        self.insert_map()
        self.draw_labels()
        self.assign_quantities()
        self.initialize_mln()

    def update_visualization(self, i):
        # Calculate sizes of links and nodes, and update the visualization of the network
        edges = self.net.edges()
        link_widths = [self.net[u][v]['weight']*self.link_size_scaling for u, v in edges]
        node_sizes = copy.deepcopy(self.quantities)
        for n in range(len(node_sizes)):
            node_sizes[n] = self.node_size_min if node_sizes[n] < self.node_size_min else node_sizes[n]
            node_sizes[n] = node_sizes[n]*self.node_size_scaling

        # Update network visualization
        # Draw links of the multilayer network
        nx.draw_networkx_edges(self.net, pos=self.all_pos, edges=edges, width=link_widths, edge_color=self.colors[0],
                               ax=self.ax_mln, arrowstyle=self.arrow, arrowsize=.5, connectionstyle='arc3,rad=0.2')
        # Draw dynamic nodes
        nx.draw_networkx_nodes(self.dynamic_nodes, pos=self.all_pos, node_size=node_sizes,
                               node_color=self.color_list[:self.nodes], ax=self.ax_nodes, with_labels=False)

        # Put links into the right z-position
        for patch in self.ax_mln.patches[self.layers:]:
            patch.zorder = 1
        # Animate the time series that depicts the aggregate amount of quantity in the system
        self.time_series_y.append(np.round(np.sum(self.quantities), 2))
        self.line.set_xdata(self.time_series_x[:i])
        self.line.set_ydata(self.time_series_y)

        plt.pause(.001)
        # Clear inter-layer links & dynamic nodes
        for artist in self.ax_mln.patches[self.layers:] + self.ax_nodes.collections:
            artist.remove()

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
        for i in range(1, self.iterations):
            self.update_visualization(i)
            self.network_dynamics()
            self.constrain_and_reassign()
            # If user stops visualization, terminate the script
            if not plt.fignum_exists(0):
                break


if __name__ == "__main__":
    # Set up & simulate the co-evolving multilayer network
    network = EuropeanSystem(layers=3, nodes=27, link_p=.05)
    network.initialize()
    network.simulate()

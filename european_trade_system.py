# -*- coding: utf-8 -*-
# Created with Python 3.7
"""
This code generates a simulation of a European economic system.
The temporal evolution of imports and exports among several countries is visualized.
"""


import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle, Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import networkx as nx
import copy
import random
import pandas as pd
import parameters as pars


class EuropeanSystem:

    def __init__(self):
        # Initialize parameters
        # Network parameters
        self.layers = pars.layers
        self.nodes = pars.nodes

        # Get data and initialize quantities and links
        self.start_at = (pars.start_sim - 2000) * pars.months
        self.end_at = (pars.end_sim - 2000) * pars.months
        self.c_vis = pars.c_vis
        self.trade_data = np.load(pars.data_path)[self.start_at:self.end_at, :, :self.nodes, :self.nodes]
        self.quantities = [np.sum(self.trade_data[0, 1, c, :]) for c in range(self.nodes)]
        self.agg_exports = np.sum(self.trade_data[:, 1, :, :], axis=2)  # since index 1 denotes exports
        self.agg_links = np.sum(self.trade_data, axis=0)
        self.links = self.trade_data[0, :, :, :]
        self.iterations = self.trade_data.shape[0]

        # Drawing parameters
        self.min_trade_size = pars.min_trade_size
        self.layer_names = ['Imports', 'Exports']
        self.node_size_min = 0.01
        self.node_font_sizes = int(12)
        self.mln_node_size = 0.3
        self.node_size_scaling = 15000
        self.link_size_scaling = 3.5
        self.arrow = ArrowStyle('simple', head_length=10, head_width=10, tail_width=.75)
        self.mln_node_sizes = self.layers*self.nodes * [self.mln_node_size*self.node_size_scaling]
        self.dynamic_node_range = range(self.layers*self.nodes, (self.layers+1)*self.nodes)
        self.colors = ['#003071', 'white', 'grey', '#731212']
        self.layer_y_dist = 17
        self.map_layer_dist = 17
        self.mln_xlims = (-4, 4)
        self.mln_ylims = (-20, 30)
        self.img_size_orig = (1160, 644)
        self.img_size = (850, 335)
        self.img_zoom = 0.72
        self.img_xy = [-0.5, -14]
        self.pll_x = [-3.1, 2.8, 3.3, -2.6]
        self.pll_y = [y + self.map_layer_dist for y in [-19.4, -19.4, -8.4, -8.4]]
        self.txt_ie_x = -4
        self.txt_x = self.mln_xlims[0]
        self.txt_y = self.mln_ylims[1]
        self.font = 'verdana'
        self.txt_font_size = 20
        self.dynamic_node_scale = 3
        self.dynamic_node_lim_x_offset = 1
        self.dynamic_node_lim_y_offset = 3
        self.ax_3_xlims = (0, self.iterations)
        self.ax_3_ylims = (0, np.max(np.sum(self.agg_exports, axis=1)) * 1.1)
        self.tick_size = 13
        self.label_font_size = 15
        self.ax_3_lw = 1

        # Set up the figure
        pylab.ion()
        self.dpi = 100
        self.fig_size = (19.2, 10.8)
        self.resolution = [f * self.dpi for f in self.fig_size]
        self.fig = plt.figure(0, figsize=self.fig_size, dpi=self.dpi)
        self.fig.canvas.set_window_title('European System')
        self.ax_mln = plt.subplot2grid((10, 10), (0, 0), rowspan=10, colspan=5)
        self.ax_nodes = plt.subplot2grid((10, 10), (0, 5), rowspan=4, colspan=5)
        self.ax_3 = plt.subplot2grid((10, 10), (4, 6), rowspan=5, colspan=3)
        self.fig.tight_layout(pad=3, h_pad=1.25, w_pad=1.25)

    def get_layouting(self):
        # Read in data related to the map
        data = pd.read_excel('data/countries.xlsx', nrows=self.nodes)
        countries = list(data['Country'].values)
        cc_list = list(data['Code'].values)
        labels = dict(zip(range(0, self.nodes), cc_list))
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

        return countries, labels, color_list, all_pos, aux_all_pos,

    def get_scalers(self):
        # Fit sigmoid functions that scale the width of the visualized nodes and links
        # Links
        link_min, link_max = np.nanmin(self.trade_data[np.nonzero(self.trade_data)]), np.nanmax(self.trade_data)
        link_s = 4/link_max
        link_c = (link_max-link_min)/2
        link_scaler = lambda x: 1/(1+np.exp(-link_s*(x-link_c)))
        # Nodes
        node_min, node_max = np.nanmin(self.agg_exports[np.nonzero(self.agg_exports)]), np.nanmax(self.agg_exports)
        node_s = 2/(node_max-node_min)
        node_c = 1.5*(node_max-node_min)
        node_scaler = lambda x: 1/(1+np.exp(-node_s*(x-node_c)))
        return node_scaler, link_scaler

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
        self.ax_3.set_xlabel('Time [months]', fontsize=self.label_font_size)
        self.ax_3.set_ylabel('Aggregate european exports [USD]', fontsize=self.label_font_size)
        self.ax_3.set_xlim(self.ax_3_xlims)
        self.ax_3.set_ylim(self.ax_3_ylims)
        self.ax_3.grid('-', alpha=.5)

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

    def links_tensor_to_matrix(self, links_tensor):
        # Convert adjacency tensor to a matrix form, for networkx compliance
        links_mat = np.zeros((self.layers*self.nodes, self.layers*self.nodes))
        for n in range(0, self.layers):
            links_mat[self.nodes*n:self.nodes*(n+1), self.nodes*n:self.nodes*(n+1)] = links_tensor[n, :, :]
        return links_mat

    def links_matrix_to_tensor(self, links_mat):
        # Convert networkx-compliant matrix form to an adjancecy tensor
        links = np.zeros((self.layers, self.nodes, self.nodes))
        for n in range(0, self.layers):
            links[n, :, :] = links_mat[self.nodes*n:self.nodes*(n+1), self.nodes*n:self.nodes*(n+1)]
        return links

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
            txt_ie_y = pll_y[0] + (pll_y[-1]-pll_y[0])/2
            self.ax_mln.text(self.txt_ie_x, txt_ie_y, self.layer_names[l], fontsize=self.txt_font_size, fontname=self.font)
        # Draw nodes
        nx.draw_networkx_nodes(self.net, pos=self.all_pos, node_soze=self.mln_node_sizes, node_color=self.color_list,
                               ax=self.ax_mln, with_labels=False)
        self.ax_mln.collections[0].zorder = 3
        # Draw markers between identical nodes in multilayer network
        nx.draw_networkx_edges(self.auxiliary_net, pos=self.aux_all_pos, edgelist=self.links_interlayer,
                               ax=self.ax_mln, style='dotted', edge_color=self.colors[2], alpha=.75, arrows=False)
        self.ax_mln.collections[1].zorder = 2

    def initialize(self):
        # A second initialization (next to __init__) for all parameters and variables that require computation
        self.countries, self.labels, self.color_list, self.all_pos, self.aux_all_pos = self.get_layouting()
        self.node_scaler, self.link_scaler = self.get_scalers()
        # Initialize the network instance from the aggregate links that are to occur during the simulation
        self.net = nx.from_numpy_matrix(self.links_tensor_to_matrix(self.agg_links), create_using=nx.DiGraph())
        self.dynamic_nodes = nx.from_numpy_matrix(np.zeros((self.nodes, self.nodes)))
        self.auxiliary_net, self.links_interlayer = self.initialize_interlayer_links()
        self.time_series_x, self.time_series_y, self.line = self.initialize_time_series()
        self.arrange_subplots()
        self.insert_map()
        self.draw_labels()
        self.assign_quantities()
        self.initialize_mln()

    def system_dynamics(self, i):
        # Co-evolving dynamics
        self.quantities = self.agg_exports[i, :]
        # Get links from data
        self.links = self.trade_data[i, :, :, :]

    def assign_quantities(self):
        # Assign quantities to nodes
        quantities_dict = dict(zip(self.dynamic_node_range, self.quantities))
        nx.set_node_attributes(self.net, quantities_dict, 'Quantities')

    def constrain_and_reassign(self):
        # Reshape and reassign both links and quantities to the network
        links_mat = self.links_tensor_to_matrix(self.links)
        # Reassign quantities to nodes and weights to links
        self.assign_quantities()
        # Reassign link weights
        for edge_triple in self.net.edges(data=True):
            u, v = edge_triple[0:2]
            # If only one country's relations shall be visualized, assign only the respective weights
            if len(self.c_vis) > 0:
                # Imports indices range from 0:self.nodes, whereas export indices range from self.nodes:2*self.nodes
                c_vis_imp, c_vis_exp = self.c_vis, [self.nodes + c for c in self.c_vis]
                edge_triple[-1]['weight'] = links_mat[u, v] if v in c_vis_imp or u in c_vis_exp else 0
            else:
                edge_triple[-1]['weight'] = links_mat[u, v]

    def update_visualization(self, i):
        # Calculate sizes of links and nodes, and update the visualization of the network
        all_edges = self.net.edges(data=True)
        # Extract only those connections for which data exists that satisfies a boundary condition
        edges = []
        for edge_triple in all_edges:
            if edge_triple[-1]['weight'] > self.min_trade_size:
                edges.append(edge_triple[:2])
        # Calculate scaling of links and dynamical nodes
        link_widths = [self.link_scaler(self.net[u][v]['weight'])*self.link_size_scaling for u, v in edges]
        node_sizes = copy.deepcopy(self.quantities)
        for n in range(len(node_sizes)):
            node_sizes[n] = self.node_scaler(node_sizes[n])
            node_sizes[n] = self.node_size_min if node_sizes[n] < self.node_size_min else node_sizes[n]
            node_sizes[n] = node_sizes[n]*self.node_size_scaling

        # Update network visualization
        # Draw links of the multilayer network
        nx.draw_networkx_edges(self.net, pos=self.all_pos, edgelist=edges, width=link_widths, edge_color=self.colors[0],
                               ax=self.ax_mln, arrowstyle=self.arrow, arrowsize=.5, connectionstyle='arc3,rad=0.2')
        # Draw dynamic nodes
        nx.draw_networkx_nodes(self.dynamic_nodes, pos=self.all_pos, node_size=node_sizes,
                               node_color=self.color_list[:self.nodes], ax=self.ax_nodes, with_labels=False)

        # Put links into the right z-position
        for patch in self.ax_mln.patches[self.layers:]:
            patch.zorder = 1
        # Animate the time series that depicts the aggregate amount of quantity in the system
        self.time_series_y.append(np.round(np.sum(self.quantities), 2))
        self.line.set_xdata(self.time_series_x[:i+1])  # +1 cause of pythonic indexing
        self.line.set_ydata(self.time_series_y)
        # Show current month and year
        txt = str("{:02d}".format((i+self.start_at)%pars.months+1)) + '/' + str(pars.years[0] + (i+self.start_at)//pars.months)
        date_text = self.ax_mln.text(self.txt_x, self.txt_y, txt, fontsize=self.txt_font_size, fontname=self.font)

        plt.pause(.01)
        # Clear inter-layer links, dynamic nodes and text
        for artist in self.ax_mln.patches[self.layers:] + self.ax_nodes.collections + [date_text]:
            artist.remove()

    def simulate(self):
        # Start simulating the dynamics and updating the visualization
        # Repeatedly run dynamics and update visualization
        for i in range(0, self.iterations):
            self.system_dynamics(i)
            self.constrain_and_reassign()
            self.update_visualization(i)
            # If user stops visualization, terminate the script
            if not plt.fignum_exists(0):
                break


if __name__ == "__main__":
    # Set up & simulate the co-evolving multilayer network
    network = EuropeanSystem()
    network.initialize()
    network.simulate()

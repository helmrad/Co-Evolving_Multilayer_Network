# Co-Evolving Multilayer Network

This code generates a co-evolving multilayer network, that is, a network with multiple types of interactions between the nodes, while both the state of the nodes and the links that connect the nodes evolve dynamically.

<img src="./images/system_screenshot.svg">

# Application Example: European Economic System

This framework can be used to describe a system in which particular entities, in this case European countries, interact with each other in several ways (thus the layers), to exchange, consume, and produce some quantity in a collective manner.
The dynamic width of the links in the multilayer network on the left depicts the strength of the interaction between the respective countries. The dynamic size of the nodes in the top right depicts the amount of accumulated quantity for each country. The plot in the bottom right shows the temporal evolution of the aggregate quantity of all countries combined.

<img src="./images/european_random_system_screenshot.svg">

As this is intended to be just an example of usage, the system dynamics are random and not based on real-world data.


Created with Python 3.7

Dependencies:
- matplotlib 3.2.1
- numpy 1.18.4
- networkx 2.4
- names 0.3.0

import networkx as nx
import matplotlib.pyplot as plt

M = nx.MultiGraph()

M.add_nodes_from(['X','Y','Z'])
M.add_edges_from([('X','Y'),('X','Y'),('Y','Z')])

nx.draw_spectral(M,with_labels=True)
plt.show()
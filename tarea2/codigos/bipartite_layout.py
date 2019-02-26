import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_nodes_from(['A', 'B', 'C'], bipartite = 0)
G.add_nodes_from(['D', 'E', 'F'], bipartite = 1)

G.add_edges_from([('A', 'D'), ('A', 'E')])
G.add_edges_from([('B', 'E')])
G.add_edges_from([('C', 'D'), ('C', 'F')])

nx.draw(G, pos = nx.bipartite_layout(G, ['A', 'B', 'C']), with_labels = True)

plt.show()
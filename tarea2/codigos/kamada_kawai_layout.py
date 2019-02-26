import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G'])

G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'A')])
G.add_edges_from([('A', 'D'), ('A', 'G')])
G.add_edges_from([('E', 'C'), ('E', 'G')])

nx.draw(G, pos = nx.kamada_kawai_layout(G), with_labels=True)

plt.show()
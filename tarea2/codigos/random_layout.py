import networkx as nx
import matplotlib.pyplot as plt

G=nx.DiGraph()

G.add_node("A")
G.add_nodes_from(["B", "C", "D", "E"])

G.add_edges_from([("A","B"), ("A","C"), ("A","D"), ("A","E")])

nx.draw(G, pos = nx.random_layout(G), with_labels = True)

plt.show()
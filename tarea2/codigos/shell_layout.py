import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_node("A")
G.add_nodes_from(["B", "C", "D", "E", "F"])
G.add_nodes_from(["G", "H", "I", "J", "K", "L"])

G.add_path(["B", "C", "D", "E", "F", "B"])
G.add_path(["G", "H", "I", "J", "K", "L", "G"])

G.add_edges_from([("H", "C"), ("J", "E"), ("D", "A"), ("F", "A")])

nx.draw(G, pos = nx.shell_layout(G), with_labels=True)

plt.show()

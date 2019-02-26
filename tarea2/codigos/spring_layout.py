import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_edges_from([("A", "B", {'myweight':1}), ("B", "C", {'myweight':2}),
                  ("A", "D", {'myweight':3}), ("B", "D", {'myweight':10})])

nx.draw_networkx(G, nx.spring_layout(G, weight = "myweight"))

plt.axis('off')
plt.show()
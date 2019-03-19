import networkx as nx
import time as tm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

G = nx.Graph()

G.add_nodes_from(['A', 'B', 'C'], bipartite = 0)
G.add_nodes_from(['D', 'E', 'F'], bipartite = 1)

G.add_edges_from([('A', 'D'), ('A', 'E')])
G.add_edges_from([('B', 'E')])
G.add_edges_from([('C', 'D'), ('C', 'F')])

nodos1 = [len(G.nodes)] * 30
edges1 = [len(G.edges)] * 30

tiempoAngel1 = []
for i in range(30):
    start = tm.time()
    for x in range(8000000):
        nx.all_shortest_paths(G, source='C', target='B')
    end = tm.time()
    tiempoAngel1.append(end - start)

media = np.mean(tiempoAngel1)
desviacion = np.std(tiempoAngel1)
p = stats.shapiro(tiempoAngel1)

text1 = "$\mu =$" + str(round(media, 3)) + "$,\sigma =$" + str(round(desviacion, 3))
text2 = "$p = $" + str(round(p[1], 3))

plt.hist(tiempoAngel1, bins='auto', alpha = 0.9, rwidth=0.85, color= 'green')
plt.grid(axis='y', alpha=1)
plt.ylabel('Frecuencia')
plt.xlabel('Tiempo (seg)')
plt.text(6.6, 8.9, text1)
plt.text(6.7,8.3, text2)



#Grafo 2
G=nx.Graph()

G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G'])

G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'A')])
G.add_edges_from([('A', 'D'), ('A', 'G')])
G.add_edges_from([('E', 'C'), ('E', 'G')])

nodos2 = [len(G.nodes)] * 30
edges2 = [len(G.edges)] * 30

tiempoAngel2 = []
for i in range(30):
    start = tm.time()
    for x in range(20000):
        nx.betweenness_centrality(G, normalized=True)
    end = tm.time()
    tiempoAngel2.append(end - start)

media = np.mean(tiempoAngel2)
desviacion = np.std(tiempoAngel2)
p = stats.shapiro(tiempoAngel2)

text1 = "$\mu =$" + str(round(media, 3)) + "$,\sigma =$" + str(round(desviacion, 3))
text2 = "$p = $" + str(round(p[1], 6))

plt.hist(tiempoAngel2, bins='auto', alpha = 0.9, rwidth=0.85, color= 'green')
plt.grid(axis='y', alpha=1)
plt.ylabel('Frecuencia')
plt.xlabel('Tiempo (seg)')
plt.text(7.4 * 400, 14.5, text2)



#Grafo 3
G=nx.Graph()

G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G'])

G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'A')])
G.add_edges_from([('A', 'D'), ('A', 'G')])
G.add_edges_from([('E', 'C'), ('E', 'G')])

nodos3 = [len(G.nodes)] * 30
edges3 = [len(G.edges)] * 30

tiempoAngel3 = []
for i in range(30):
    start = tm.time()
    for x in range(100000):
        nx.dfs_tree(G, "D")
    end = tm.time()
    tiempoAngel3.append(end - start)

media = np.mean(tiempoAngel3)
desviacion = np.std(tiempoAngel3)
p = stats.shapiro(tiempoAngel3)

text1 = "$\mu =$" + str(round(media, 3)) + "$,\sigma =$" + str(round(desviacion, 3))
text2 = "$p = $" + str(round(p[1], 5))

plt.hist(tiempoAngel3, bins='auto', alpha = 0.9, rwidth=0.85, color= 'green')
plt.grid(axis='y', alpha=1)
plt.ylabel('Frecuencia')
plt.xlabel('Tiempo (seg)')
plt.text(7.02 * 80,6.5, text2)


#Grafo 4
G = nx.DiGraph()

G.add_node("A")
G.add_nodes_from(["B", "C", "D", "E", "F"])
G.add_nodes_from(["G", "H", "I", "J", "K", "L"])

G.add_path(["B", "C", "D", "E", "F", "B"])
G.add_path(["G", "H", "I", "J", "K", "L", "G"])

G.add_edges_from([("H", "C"), ("J", "E"), ("D", "A"), ("F", "A")])

nodos4 = [len(G.nodes)] * 30
edges4 = [len(G.edges)] * 30

tiempoAngel4 = []
for i in range(30):
    start = tm.time()
    for x in range(100000):
        nx.greedy_color(G, strategy='largest_first')
    end = tm.time()
    tiempoAngel4.append(end - start)

media = np.mean(tiempoAngel4)
desviacion = np.std(tiempoAngel4)
p = stats.shapiro(tiempoAngel4)

text1 = "$\mu =$" + str(round(media, 3)) + "$,\sigma =$" + str(round(desviacion, 3))
text2 = "$p = $" + str(round(p[1], 5))

plt.hist(tiempoAngel4, bins='auto', alpha = 0.9, rwidth=0.85, color= 'green')
plt.grid(axis='y', alpha=1)
plt.ylabel('Frecuencia')
plt.xlabel('Tiempo (seg)')
plt.text(7.8 * 80, 14.8, text2)


#Grafo 5
G=nx.Graph()

G.add_node("A")
G.add_nodes_from(["B", "C", "D", "E"])

G.add_edges_from([("A","B"), ("A","C"), ("A","D"), ("A","E")])

nodos5 = [len(G.nodes)] * 30
edges5 = [len(G.edges)] * 30

tiempoAngel5 = []
for i in range(30):
    start = tm.time()
    for x in range(25000):
        nx.max_weight_matching(G)
    end = tm.time()
    tiempoAngel5.append(end - start)

media = np.mean(tiempoAngel5)
desviacion = np.std(tiempoAngel5)
p = stats.shapiro(tiempoAngel5)

text1 = "$\mu =$" + str(round(media, 3)) + "$,\sigma =$" + str(round(desviacion, 3))
text2 = "$p = $" + str(round(p[1], 3))

plt.hist(tiempoAngel5, bins='auto', alpha = 0.9, rwidth=0.85, color= 'green')
plt.grid(axis='y', alpha=1)
plt.ylabel('Frecuencia')
plt.xlabel('Tiempo (seg)')
plt.text(6.6, 8.9, text1)
plt.text(5.1 * 320, 9, text2)

nodos = nodos1 + nodos2 + nodos3 + nodos4 + nodos5
edges = edges1 + edges2 + edges3 + edges4 + edges5

datos = {'Nodos' : nodos, 'Aristas' : edges, 'Tiempos' : tiempos}

dt = pd.DataFrame(datos, columns = ['Nodos', 'Aristas', 'Tiempos'])

colores = ["green"] * 30 + ["red"] * 30 + ["blue"] * 30 + ["yellow"] * 30 + ["pink"] * 30
grupos = ["Caminos cortos"] * 30 + ["Nodos centrales"] * 30 + ["DFS"] * 30 + ["Colores"] * 30 + ["Maximo peso"] * 30
formas = ["s"] * 30 + ["^"] * 30 + ["p"] * 30 + ["*"] * 30 + ["d"] * 30

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for tiempo, nodo, color, grupo, forma in zip(tiempos, nodos, colores, grupos, formas):
    ax.scatter(tiempo, nodo, alpha=0.8, c=color, edgecolors='none', s=30, label=grupo, marker = forma)

plt.legend(loc=2)

plt.scatter(dt.Tiempos, dt.Nodos, c = dt.Aristas)
plt.xlabel("Tiempo (seg)")
plt.ylabel("Cantidad de nodos")

plt.scatter(dt.Tiempos, dt.Aristas, c = dt.Aristas)
plt.xlabel("Tiempo (seg)")
plt.ylabel("Cantidad de aristas")
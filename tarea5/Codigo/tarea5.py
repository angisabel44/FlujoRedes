import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rd
import time as tm
import pandas as pd
import seaborn as sns

numerosdenodos = 16

datos = pd.DataFrame()
caracteristicas = pd.DataFrame()

for g in range(5):
	G = nx.generators.classic.wheel_graph(numerosdenodos)

	for k in G.edges():
	    G[k[0]][k[1]]['weight'] = np.random.normal(10, 1)

	st = rd.sample(range(len(G)), 2)
	grafo = 'G' + str(g + 1)

	for i in range(30):
		t1 = tm.time();
		for j in range(25):
			flowmax, flowdict = nx.maximum_flow(G, st[0], st[1], capacity='weight')
		t1 = tm.time() - t1
		dato = pd.DataFrame({'Grafo': [grafo], 'Tiempo': 100 * t1, 'Flujo': flowmax})
		datos = datos.append(dato)

	grado = nx.degree(G)
	grado['Grafo'] = grafo
	grado['Caracteristica'] = 'Grado distribucion'
	dato = pd.DataFrame([grado])
	caracteristicas = caracteristicas.append(dato)

	clustercoef = nx.clustering(G)
	clustercoef['Grafo'] = grafo
	clustercoef['Caracteristica'] = 'Coeficient clustering'
	dato = pd.DataFrame([clustercoef])
	caracteristicas = caracteristicas.append(dato)

	closcent = nx.closeness_centrality(G)
	closcent['Grafo'] = grafo
	closcent['Caracteristica'] = 'Closeness centrality'
	dato = pd.DataFrame([closcent])
	caracteristicas = caracteristicas.append(dato)

	loadcent = nx.load_centrality(G)
	loadcent['Grafo'] = grafo
	loadcent['Caracteristica'] = 'Load centrality'
	dato = pd.DataFrame([loadcent])
	caracteristicas = caracteristicas.append(dato)

	ecce = nx.eccentricity(G)
	ecce['Grafo'] = grafo
	ecce['Caracteristica'] = 'Excentricidad'
	dato = pd.DataFrame([ecce])
	caracteristicas = caracteristicas.append(dato)

	pag = nx.pagerank(G)
	pag['Grafo'] = grafo
	pag['Caracteristica'] = 'Page rank'
	dato = pd.DataFrame([pag])
	caracteristicas = caracteristicas.append(dato)


	pos = nx.spectral_layout(G)

	nodos = []
	pesos = []
	coloresnodos = []
	coloresaristas = []

	for x in G.nodes():
		nodos.append(x + 1)
		if x == st[0]:
			coloresnodos.append('g')
		elif x == st[1]:
			coloresnodos.append('r')
		else:
			coloresnodos.append('b')

	for u, v in G.edges():
		pesos.append(0.20 * G[u][v]['weight'])
		if flowdict[u][v] == 0 and flowdict[v][u] == 0:
			coloresaristas.append('b')
		else:
			coloresaristas.append('r')

	nx.draw_networkx_nodes(G, pos, node_color = coloresnodos)
	nx.draw_networkx_labels(G, pos, label = nodos, font_size = 7)
	nx.draw_networkx_edges(G, pos, edge_color = coloresaristas, width = pesos)
	plt.savefig(grafo + '.eps')
	plt.clf()

#datos.to_csv(r'datos.csv')
#caracteristicas.to_csv(r'caracteristicas.csv')

sns.boxplot(x = 'Grafo', y = 'Tiempo', data = datos)
plt.xlabel('Generador de grafos')
plt.ylabel('Tiempo (milisegndos)')
plt.savefig('figura1.eps')
plt.clf()

sns.boxplot(x = 'Grafo', y = 'Flujo', data = datos)
plt.xlabel('Generador de grafos')
plt.ylabel('Flujo maximo')
plt.savefig('figura2.eps')
plt.clf()
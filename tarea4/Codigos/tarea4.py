import networkx as nx
import pandas as pd
import numpy as np
import time as tm
import random as rd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

from networkx.algorithms.flow import shortest_augmenting_path

datos = pd.DataFrame()

tamanos = [32, 64, 128, 256]
niu = 10
sigma = 1

for i in tamanos:
	for j in range(10):
		#PRIMER GRAFO
		tiempo1 = tm.time()
		G = nx.complete_graph(i)

		for k in G.edges():
			G[k[0]][k[1]]['weight'] = np.random.normal(niu, sigma)

		tiempo1 = tm.time() - tiempo1

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				nx.maximum_flow_value(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo completo'], 'Algoritmo': ['Maximo flujo'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				shortest_augmenting_path(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo completo'], 'Algoritmo': ['Camino aumento corto'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				nx.algorithms.flow.preflow_push(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo completo'], 'Algoritmo': ['Empujo preflujo'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

		#SEGUNDO GRAFO
		tiempo1 = tm.time()
		G = nx.generators.classic.circulant_graph(i, [1, 2])

		for k in G.edges():
			G[k[0]][k[1]]['weight'] = np.random.normal(niu, sigma)

		tiempo1 = tm.time() - tiempo1

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				nx.maximum_flow_value(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo circular'], 'Algoritmo': ['Maximo flujo'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				shortest_augmenting_path(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo circular'], 'Algoritmo': ['Camino aumento corto'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				nx.algorithms.flow.preflow_push(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo circular'], 'Algoritmo': ['Empujo preflujo'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

		#TERCER GRAFO
		tiempo1 = tm.time()
		G = nx.generators.classic.wheel_graph(i)

		for k in G.edges():
			G[k[0]][k[1]]['weight'] = np.random.normal(niu, sigma)

		tiempo1 = tm.time() - tiempo1

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				nx.maximum_flow_value(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo rueda'], 'Algoritmo': ['Maximo flujo'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				shortest_augmenting_path(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo rueda'], 'Algoritmo': ['Camino aumento corto'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

		for k in range(5):
			st = rd.sample(range(len(G)), 2)
			tiempo2 = tm.time()
			for l in range(5):
				nx.algorithms.flow.preflow_push(G, st[0], st[1], capacity = 'weight')
			tiempo2 = tm.time() - tiempo2 + tiempo1

			dato = pd.DataFrame({'Generador': ['Grafo rueda'], 'Algoritmo': ['Empujo preflujo'], 'Orden': len(G), 'Densidad': (G.size()/nx.complete_graph(i).size()), 'Tiempo': tiempo2*100})
			datos = datos.append(dato)

sns.boxplot(x = 'Generador', y = 'Tiempo', hue = 'Algoritmo', data = datos)
plt.xlabel('Generador de grafos')
plt.ylabel('Tiempo (milisegndos)')
plt.savefig('figura1.eps')

sns.boxplot(x = 'Algoritmo', y = 'Tiempo', hue = 'Generador', data = datos)
plt.xlabel('Algoritmo de maximo flujo')
plt.ylabel('Tiempo (milisegndos)')
plt.savefig('figura2.eps')

sns.boxplot(x = 'Orden', y = 'Tiempo', hue = 'Generador', data = datos)
plt.xlabel('Ordenes de los grafos')
plt.ylabel('Tiempo (milisegndos)')
plt.savefig('figura3.eps')

sns.boxplot(x = 'Densidad', y = 'Tiempo', data = datos)
plt.xlabel('Densidad de los grafos')
plt.ylabel('Tiempo (milisegndos)')
plt.savefig('figura4.eps')

model= ols('Tiempo ~ Generador + Orden + Algoritmo + Densidad + Generador*Orden + Generador*Algoritmo + Generador*Densidad + Orden*Algoritmo + Orden*Densidad + Algoritmo*Densidad', 
	data=datos).fit()

ANOVA = sm.stats.anova_lm(model, typ=2)
print(ANOVA)
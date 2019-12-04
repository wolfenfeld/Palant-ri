---
layout: page

title: Graphs Visualisations

path: /get_started/graphs

categories: [get_started] 
---
```python
import networkx as nx
karate_club_graph = nx.karate_club_graph()
```
```python
from palantiri.GraphPlotHandlers import GraphPlotHandler
```

```python
plot_handler = GraphPlotHandler(karate_club_graph,node_data_key='club')
plot_handler.plot_graph()
```
{% include scripts/plots/karate_graph.html %}


```python
from sklearn.datasets import load_wine

data = load_wine()

X = data.data

y = data.target

feature_names = data.feature_names
```

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X, y)

plot_handler = DecisionTreePlotHandler(decision_tree=tree,feature_names=feature_names)
```

```python
plot_handler.plot_graph()
```

{% include scripts/plots/decision_tree_graph.html %}

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=15,
    max_depth=7,
    random_state=42
)

forest.fit(X, y)

plot_handler = RandomForestPlotHandler(random_forest=forest, feature_names=feature_names)
```

```python
plot_handler.plot_graph(layout)
```
{% include scripts/plots/random_forest_graph.html %}


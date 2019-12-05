---
layout: page

title: Graphs Visualisations

path: /get_started/graphs

categories: [get_started] 
---


In this tutorial we will show how to use the Palantíri's out of the box visualizations tools on *network-x* graphs and on *sklearn* decision trees and random forests.


## Network-x Graphs

We will start with a *network-x* graph.

We will use the "Karate Club" data set from the *network-x* package:
```python
import networkx as nx
karate_club_graph = nx.karate_club_graph()
```

For this example we will use the *GraphPlotHandler*
```python
from palantiri.GraphPlotHandlers import GraphPlotHandler
```

When initializing the *GraphPlotHandler* with the graph object and a *node_data_key* which holds the data that we wish to display for each node.

Once the plot handler is initialized, the only thing that is left is plotting the graph:

```python
plot_handler = GraphPlotHandler(karate_club_graph,node_data_key='club')
plot_handler.plot_graph()
```
{% include scripts/plots/karate_graph.html %}

## Decision Tree Graphs

Another capability of Palantíri is to plot the structure of a decision tree.

In the following examples we will use the "wine" data set, provided by *sklearn*:

```python
from sklearn.datasets import load_wine

data = load_wine()

X = data.data

y = data.target

feature_names = data.feature_names
```
We will use the *DecisionTreePlotHandler* to plot the *DecisionTreeClassifier* topology.

First we will initialize the plot handler with the decision tree trained classifier: 
```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X, y)

plot_handler = DecisionTreePlotHandler(decision_tree=tree,feature_names=feature_names)
```
Once the *DecisionTreePlotHandler* is initialized we can plot:
```python
plot_handler.plot_graph()
```

{% include scripts/plots/decision_tree_graph.html %}

## Random Forest Graphs

In the last example we will use the same wine data set and plot *sklearn*'s trained *RandomForestClassifier*. 

As in the previous example, all we need to do is initialize the *RandomForestPlotHandler* with the trained classifier:
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
And again, once the *RandomForestPlotHandler* is initialized, all that is left is to plot:
```python
plot_handler.plot_graph()
```
{% include scripts/plots/random_forest_graph.html %}


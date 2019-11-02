---
layout: post
title: Clustering Visualisations
categories: [get_started] 
---
# Clustering Models
In this tutorial we will show how to visualize a trained clustering model (Scikit Learn) and display the data points.

First we will import the relevant packages:
```python
from sklearn.cluster import KMeans,AffinityPropagation
from sklearn.datasets import load_iris, load_breast_cancer
```

We will start with a *Two Dimensional Plot Handler*.

We will import the sample data set and the chosen clustering model:
```python
iris = load_iris()

iris_cluster = AffinityPropagation()
iris_cluster.fit(iris.data[:,:2]);
```
We will now initialize the plot handler with the trained model and the sample data-set: 
```python
from palantiri.ClusteringPlotHandlers import TwoDimensionalClusteringPlotHandler
plot_handler = TwoDimensionalClusteringPlotHandler(dataset=iris,trained_cluster=iris_cluster)
```
Once everything is set up we can start with the visulizations:
```python
plot_handler.plot_prediction()
```

{% include scripts/plots/2d-clutering-plot.html %}

We will now show how to use the *Three Dimensional Plot Handler*.

Importing the data and training the clustering model:
```python
iris = load_iris()

iris_cluster = KMeans(n_clusters=3)
iris_cluster.fit(iris.data[:,:3]);
```

Initializing the plot handler with the trained model and the data points:
```python
from palantiri.ClusteringPlotHandlers import ThreeDimensionalClusteringPlotHandler
plot_handler = ThreeDimensionalClusteringPlotHandler(dataset=iris,trained_cluster=iris_cluster)
```
Once everything is setup we can plot the prediction:

```python
plot_handler.plot_prediction()
```

{% include scripts/plots/3d-clutering-plot.html %}

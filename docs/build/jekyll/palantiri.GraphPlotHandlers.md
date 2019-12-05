---
date: '2019-12-05T12:44:50.088Z'
docname: palantiri.GraphPlotHandlers
images: {}
path: /palantiri-graph-plot-handlers
title: palantiri.GraphPlotHandlers module
---

# palantiri.GraphPlotHandlers module


#### class palantiri.GraphPlotHandlers.DecisionTreePlotHandler(decision_tree, feature_names, \*\*params)
Bases: `palantiri.GraphPlotHandlers.GraphPlotHandler`

The decision tree plot handler - handles all the decision tree related plots.


#### class palantiri.GraphPlotHandlers.GraphPlotHandler(graph, node_data_key=None, \*\*params)
Bases: `palantiri.BasePlotHandlers.PlotHandler`

The graph plot handler - handles all the graph related plots.


#### build_graph_figure(figure_layout=Layout(), pos=None)
Building the graph plot figure.
:param figure_layout: a plot.ly layout object.
:param pos: the position of each node - if none: spring layout is used for position - {node_number:[x,y]}.


#### graph()
Getter for the graph
:return: the graph


#### plot_graph(figure_layout=None)
Plotting the graph figure.
:param figure_layout: a plot.ly layout object.


#### class palantiri.GraphPlotHandlers.RandomForestPlotHandler(random_forest, feature_names, \*\*params)
Bases: `palantiri.GraphPlotHandlers.GraphPlotHandler`

The random forest plot handler - handles all the random forest related plots.


#### build_graph_figure(figure_layout=Layout(), pos=None)
Building the graph plot figure.
:param figure_layout: a plot.ly layout object.
:param pos: place holder


#### plot_graph(figure_layout=None)
Plotting the random forest graph figure.
:param figure_layout: a plot.ly layout object.

---
date: '2019-12-05T12:44:50.088Z'
docname: palantiri.ClusteringPlotHandlers
images: {}
path: /palantiri-clustering-plot-handlers
title: palantiri.ClusteringPlotHandlers module
---


### class palantiri.ClusteringPlotHandlers.ClusteringPlotHandler(dataset, trained_cluster, \*\*params)
Bases: `palantiri.BasePlotHandlers.PlotHandler`

Handles all the plots related of the chosen cluster.


#### build_prediction_figure(figure_layout)
Building the classifier prediction figure.

:param figure_layout: figure layout - plot.ly Layout object.


#### dataset()
The dataset

:return: The dataset as a dictionary


#### classmethod from_pandas_dataframe(dataframe, trained_cluster, \*\*params)
Constructing the handler from a pandas dataframe.

:param dataframe: the dataframe form which the handler is constructed.

:param trained_cluster: sklearn cluster (trained / fitted).

:param params: other params.

:return: returns the classifier plot handler object.


#### plot_prediction(figure_layout=None)
Plotting the prediction figure with plot.ly’s iplot function.

:param figure_layout: figure layout - plot.ly Layout object.


#### save_prediction_figure(file_name)
Saving the prediction figure as an html file.

:param file_name: the html file name.


#### trained_cluster()
The trained cluster.

:return: The cluster in the sklearn format.


### class palantiri.ClusteringPlotHandlers.ThreeDimensionalClusteringPlotHandler(dataset, trained_cluster, \*\*params)
Bases: `palantiri.ClusteringPlotHandlers.ClusteringPlotHandler`

Handles all the plots related of the chosen cluster on 3D.


#### build_prediction_figure(figure_layout=Layout())
Plotting the cluster prediction and saving the figure.

:param figure_layout: figure layout - plot.ly Layout object.


### class palantiri.ClusteringPlotHandlers.TwoDimensionalClusteringPlotHandler(dataset, trained_cluster, \*\*params)
Bases: `palantiri.ClusteringPlotHandlers.ClusteringPlotHandler`

Handles all the plots related of the chosen cluster on 2D.


#### build_prediction_figure(figure_layout=Layout(), step_size=0.01)
Building the classifier prediction figure.

:param figure_layout: figure layout - plot.ly Layout object.

:param step_size: Plot resolution.

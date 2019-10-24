---
# This file exists to create a link to an external site in the sidebar.
# You can change it to link to a different site, or delete it to remove the link.
# To create links to social media sites, check out `_data/authors.yml`!
title: Documentation
layout: list
menu: true
order: 2
---

## Palantiri Package Modules

* * *

## palantiri.BasePlotHandlers module


### class palantiri.BasePlotHandlers.PlotHandler(\*\*params)
Bases: `object`

Base class for the plot handlers

#### static build_hexbin_figure(x, y)

#### static save_figure(figure, file_name)
Saving the figure as an html file.
:param figure: the figure to be saved.
:param file_name: The html file name

## palantiri.ClassificationPlotHandlers module


### class palantiri.ClassificationPlotHandlers.ClassifierPlotHandler(dataset, trained_classifier, \*\*params)
Bases: `palantiri.BasePlotHandlers.PlotHandler`

Handles all the plots related of the chosen classifier.


#### build_confusion_matrix(normalize=False)
Building the confusion matrix
:param normalize: if True confusion matrix is normalized.


#### build_confusion_matrix_figure(figure_layout)
Builds the confusion matrix figure in confusion_matrix_figure.
:param figure_layout: figure layout - plot.ly layout object.


#### build_prediction_figure(figure_layout)
Building the classifier prediction figure.
:param figure_layout: figure layout - plot.ly Layout object.


#### build_roc_figure(figure_layout=Layout())
Building the ROC curve figure of the classifier.
:param figure_layout: figure layout - plot.ly layout object.


#### confusion_matrix()
The confusion matrix.
:return: The confusion matrix as a numpy array.


#### dataset()
The dataset
:return: The dataset as a dictionary


### classmethod from_pandas_dataframe(dataframe, trained_classifier, \*\*params)
Constructing the handler from a pandas dataframe.
:param dataframe: the dataframe form which the handler is constructed.
:param trained_classifier: sklearn classifier (trained / fitted).
:param params: other params.
:return: returns the classifier plot handler object.


#### n_classes()
The number of classes.
:return:  An int representing the number of classes.


#### plot_confusion_matrix(figure_layout=None)
Plotting the confusion matrix figure with plot.ly’s iplot function.
:param figure_layout: figure layout - plot.ly layout object.


#### plot_prediction(figure_layout=None)
Plotting the prediction figure with plot.ly’s iplot function.
:param figure_layout: figure layout - plot.ly Layout object.


#### plot_roc(figure_layout=None)
Plotting the ROC curve figure with plot.ly’s iplot function.
:param figure_layout: figure layout - plot.ly Layout object.


#### predicted_target_score()
The predicted score - available if classifier has the predict_proba functionality.
:return: The predicted score.


#### save_confusion_matrix_figure(file_name)
Saving the confusion matrix figure as an html file.
:param file_name: the html file name.


#### save_prediction_figure(file_name)
Saving the prediction figure as an html file.
:param file_name: the html file name.


#### save_roc_figure(file_name)
Saving the ROC curve figure as an html file.
:param file_name: the html file name.


#### trained_classifier()
The trained classifier .
:return: The classifier in the sklearn format.


### class palantiri.ClassificationPlotHandlers.ThreeDimensionalClassifierPlotHandler(dataset, trained_classifier, \*\*params)
Bases: `palantiri.ClassificationPlotHandlers.ClassifierPlotHandler`

Handles all the plots related of the chosen classifier on 3D.


#### build_prediction_figure(figure_layout=Layout())
Plotting the classifier prediction and saving the figure.
:param figure_layout: figure layout - plot.ly Layout object.


### class palantiri.ClassificationPlotHandlers.TwoDimensionalClassifierPlotHandler(dataset, trained_classifier, \*\*params)
Bases: `palantiri.ClassificationPlotHandlers.ClassifierPlotHandler`

Handles all the plots related of the chosen classifier on 2D.


#### build_prediction_figure(figure_layout=Layout(), step_size=0.01)
Building the classifier prediction figure.
:param figure_layout: figure layout - plot.ly Layout object.
:param step_size: Plot resolution.

* * *

## palantiri.ClusteringPlotHandlers module


### class palantiri.ClusteringPlotHandlers.ClusteringPlotHandler(dataset, trained_cluster, \*\*params)
Bases: `palantiri.BasePlotHandlers.PlotHandler`

Handles all the plots related of the chosen cluster.


#### build_prediction_figure(figure_layout)
Building the classifier prediction figure.
:param figure_layout: figure layout - plot.ly Layout object.


#### dataset()
The dataset
:return: The dataset as a dictionary


### classmethod from_pandas_dataframe(dataframe, trained_cluster, \*\*params)
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

* * *

## palantiri.GraphPlotHandlers module


### class palantiri.GraphPlotHandlers.DecisionTreePlotHandler(decision_tree, feature_names, \*\*params)
Bases: `palantiri.GraphPlotHandlers.GraphPlotHandler`

The decision tree plot handler - handles all the decision tree related plots.


### class palantiri.GraphPlotHandlers.GraphPlotHandler(graph, node_data_key=None, \*\*params)
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


### class palantiri.GraphPlotHandlers.RandomForestPlotHandler(random_forest, feature_names, \*\*params)
Bases: `palantiri.GraphPlotHandlers.GraphPlotHandler`

The random forest plot handler - handles all the random forest related plots.


#### build_graph_figure(figure_layout=Layout(), pos=None)
Building the graph plot figure.
:param figure_layout: a plot.ly layout object.
:param pos: place holder

* * *

## palantiri.RegressionPlotHandlers module


### class palantiri.RegressionPlotHandlers.OneDimensionalRegressionPlotHandler(dataset, trained_regressor, \*\*params)
Bases: `palantiri.RegressionPlotHandlers.RegressionPlotHandler`

Handles all the plots related of the chosen 1D regression.


#### build_prediction_figure(figure_layout, step_size=0.1, x_range=None)
Building the regression figure.
:param figure_layout: figure layout - plot.ly layout object.
:param step_size: resolution of the x-axis.
:param x_range: the range of the prediction (x-axis),
list of 2 numbers - indicating the start and end of the range
if none will take the minimum and maximum of the data set.


### class palantiri.RegressionPlotHandlers.RegressionPlotHandler(dataset, trained_regressor, \*\*params)
Bases: `palantiri.BasePlotHandlers.PlotHandler`

Handles all the plots related of the chosen regressor.


#### build_prediction_figure(figure_layout)
Building the regression figure.
:param figure_layout: figure layout - plot.ly layout object.


#### dataset()
The dataset
:return: The dataset as a dictionary


### classmethod from_pandas_dataframe(dataframe, trained_regressor, \*\*params)
Constructing the handler from a pandas dataframe.
:param dataframe: the dataframe form which the handler is constructed.
:param trained_regressor: sklearn regressor (trained / fitted).
:param params: other params.
:return: returns the classifier plot handler object.


#### plot_prediction(figure_layout=None)
Plotting the regression figure with plot.ly’s iplot function.
:param figure_layout: figure layout - plot.ly layout object.


#### save_prediction_figure(file_name)
Saving the prediction figure as an html file.
:param file_name: the html file name.


#### trained_regressor()
The trained regressor.
:return: The regressor in sklearn format.


### class palantiri.RegressionPlotHandlers.TwoDimensionalRegressionPlotHandler(dataset, trained_regressor, \*\*params)
Bases: `palantiri.RegressionPlotHandlers.RegressionPlotHandler`

Handles all the plots related of the chosen regressor on 2D.


#### build_prediction_figure(figure_layout=Layout(), x_range=None, y_range=None, step_size=0.1)
Building the regression figure.
:param figure_layout: figure layout - plot.ly layout object.
:param step_size: resolution of the x-axis.
:param x_range: the range of the prediction (x-axis),
list of 2 numbers - indicating the start and end of the range
if none will take the minimum and maximum of the data set.
:param y_range: similar to x_range for the y-axis.

* * *

## palantiri.RegretPlotHandler module


### class palantiri.RegretPlotHandler.RegretPlotHandler(cumulative_regret_data, \*\*params)
Bases: `palantiri.BasePlotHandlers.PlotHandler`

The regret plot handler - handles all the regret related plots.


#### build_graph_figure(figure_layout=Layout())
Building the regret plot figure.
:param figure_layout: a plot.ly layout object.


#### plot_regret(figure_layout=None)
Plotting the graph figure.
:param figure_layout: a plot.ly layout object.

## Module contents

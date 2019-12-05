---
date: '2019-12-05T12:44:50.088Z'
docname: palantiri.RegressionPlotHandlers
images: {}
path: /palantiri-regression-plot-handlers
title: palantiri.RegressionPlotHandlers module
---

# palantiri.RegressionPlotHandlers module


#### class palantiri.RegressionPlotHandlers.OneDimensionalRegressionPlotHandler(dataset, trained_regressor, \*\*params)
Bases: `palantiri.RegressionPlotHandlers.RegressionPlotHandler`

Handles all the plots related of the chosen 1D regression.


#### build_prediction_figure(figure_layout, step_size=0.1, x_range=None)
Building the regression figure.
:param figure_layout: figure layout - plot.ly layout object.
:param step_size: resolution of the x-axis.
:param x_range: the range of the prediction (x-axis),
list of 2 numbers - indicating the start and end of the range
if none will take the minimum and maximum of the data set.


#### class palantiri.RegressionPlotHandlers.RegressionPlotHandler(dataset, trained_regressor, \*\*params)
Bases: `palantiri.BasePlotHandlers.PlotHandler`

Handles all the plots related of the chosen regressor.


#### build_prediction_figure(figure_layout)
Building the regression figure.
:param figure_layout: figure layout - plot.ly layout object.


#### dataset()
The dataset
:return: The dataset as a dictionary


#### classmethod from_pandas_dataframe(dataframe, trained_regressor, \*\*params)
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


#### class palantiri.RegressionPlotHandlers.TwoDimensionalRegressionPlotHandler(dataset, trained_regressor, \*\*params)
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

---
date: '2019-11-20T19:50:34.239Z'
docname: palantiri.BasePlotHandlers
images: {}
path: /palantiri-base-plot-handlers
title: palantiri.BasePlotHandlers module
---

# palantiri.BasePlotHandlers module


#### class palantiri.BasePlotHandlers.PlotHandler(\*\*params)
Bases: `object`

Base class for the plot handlers


#### static build_hexbin_figure(x, y)
Building a hexabin plotly graphical object figure.
:param x: array containing all the x coordinates of the selected data points.
:param y: array containing all the y coordinates of the selected data points.
:return: hexabing plotly graphical object


#### static save_figure(figure, file_name)
Saving the figure as an html file.
:param figure: plot.ly figure graphical object to be saved.
:param file_name: The html file name

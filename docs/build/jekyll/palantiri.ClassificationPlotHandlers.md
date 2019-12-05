---
date: '2019-12-05T12:44:50.088Z'
docname: palantiri.ClassificationPlotHandlers
images: {}
path: /palantiri-classification-plot-handlers
title: palantiri.ClassificationPlotHandlers module
---

# palantiri.ClassificationPlotHandlers module


#### class palantiri.ClassificationPlotHandlers.ClassifierPlotHandler(dataset, trained_classifier, \*\*params)
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


#### classmethod from_pandas_dataframe(dataframe, trained_classifier, \*\*params)
Constructing the handler from a pandas dataframe.
:param dataframe: the dataframe form which the handler is constructed.
The ‘target’ column  should be included in the dataframe.
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


#### class palantiri.ClassificationPlotHandlers.ThreeDimensionalClassifierPlotHandler(dataset, trained_classifier, \*\*params)
Bases: `palantiri.ClassificationPlotHandlers.ClassifierPlotHandler`

Handles all the plots related of the chosen classifier on 3D.


#### build_prediction_figure(figure_layout=Layout())
Plotting the classifier prediction and saving the figure.
:param figure_layout: figure layout - plot.ly Layout object.


#### class palantiri.ClassificationPlotHandlers.TwoDimensionalClassifierPlotHandler(dataset, trained_classifier, \*\*params)
Bases: `palantiri.ClassificationPlotHandlers.ClassifierPlotHandler`

Handles all the plots related of the chosen classifier on 2D.


#### build_prediction_figure(figure_layout=Layout(), step_size=0.01)
Building the classifier prediction figure.
:param figure_layout: figure layout - plot.ly Layout object.
:param step_size: Plot resolution.

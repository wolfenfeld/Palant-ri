from palantiri.BasePlotHandlers import PlotHandler

import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, auc


import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot


class ClassifierPlotHandler(PlotHandler):
    """ Handles all the plots related of the chosen classifier. """

    def __init__(self, dataset, trained_classifier, **params):
        """
        Initialization function
        :param dataset: the dataset in a dict format with the following keys:
                        'data' - numpy array with all the data points.
                        'target' - the label of the corresponding data point.
                        'target_names' - the label name.

        :param trained_classifier: sklearn classifier (trained / fitted).
                                   In order to plot the ROC plot - the classifier should have the predict_proba ability.
        :param params: other params
        """

        self._dataset = dataset
        self._trained_classifier = trained_classifier

        self._n_classes = len(set(dataset['target']))

        if hasattr(self._dataset, 'target_names'):
            self.class_names = self._dataset['target_names']
        else:
            self.class_names = ['Class {0}'.format(i) for i in range(self.n_classes)]

        # Score of the predicted target store.
        if hasattr(self._trained_classifier, 'predict_proba'):
            self._predicted_target_score = self._trained_classifier.predict_proba(self._dataset['data'])
        else:
            self._predicted_target_score = None

        self._confusion_matrix = None

        self.prediction_figure = None
        self.roc_figure = None
        self.confusion_matrix_figure = None

        super(ClassifierPlotHandler, self).__init__(**params)

    @classmethod
    def from_pandas_dataframe(cls, dataframe, trained_classifier, **params):
        """
        Constructing the handler from a pandas dataframe.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_classifier: sklearn classifier (trained / fitted).
        :param params: other params.
        :return: returns the classifier plot handler object.
        """

        assert 'target' in dataframe.columns.values, 'target values not in dataframe'

        dataset = dict()
        dataset['data'] = dataframe.drop('target', axis=1).values
        dataset['target'] = dataframe['target'].values
        dataset['feature_names'] = dataframe.drop('target', axis=1).columns.values
        return cls(dataset, trained_classifier, **params)

    @property
    def trained_classifier(self):
        """
        The trained classifier .
        :return: The classifier in the sklearn format.
        """
        return self._trained_classifier

    @property
    def dataset(self):
        """
        The dataset
        :return: The dataset as a dictionary
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        """
        The dataset setter.
        :param dataset: the new dataset
        """
        self._dataset = dataset

    @property
    def predicted_target_score(self):
        """
        The predicted score - available if classifier has the predict_proba functionality.
        :return: The predicted score.
        """
        return self._predicted_target_score

    @property
    def confusion_matrix(self):
        """
        The confusion matrix.
        :return: The confusion matrix as a numpy array.
        """
        return self._confusion_matrix

    @property
    def n_classes(self):
        """
        The number of classes.
        :return:  An int representing the number of classes.
        """
        return self._n_classes

    def build_confusion_matrix_figure(self,
                                      title='Confusion Matrix',
                                      normalize=False,
                                      colorscale='Viridis'):
        """
        Builds the confusion matrix figure in confusion_matrix_figure.
        :param title: the title of the figure
        :param normalize: if True the confusion matrix is normalized.
        :param colorscale: the color scale of the figure.
        """

        prediction = self.trained_classifier.predict(self._dataset['data'])

        self._confusion_matrix = confusion_matrix(self._dataset['target'], prediction)

        if normalize:
            cm = self._confusion_matrix.astype('float') / self._confusion_matrix.sum(axis=1)[:, np.newaxis]
        else:
            cm = self._confusion_matrix

        cm = np.flipud(cm)
        x = list(self.class_names)
        y = list(reversed(self.class_names))

        self.confusion_matrix_figure = ff.create_annotated_heatmap(z=cm, x=x, y=y, colorscale=colorscale)
        self.confusion_matrix_figure['layout']['yaxis'].update({'title': 'True Value'})
        self.confusion_matrix_figure['layout']['xaxis'].update({'title': title+'<br> <br> Predicted Value'})

    def plot_confusion_matrix(self,
                              normalize=False,
                              colorscale='Viridis'):
        """
        Plotting the confusion matrix figure with plotly's iplot function.
        If the figure is yet to be built or the default params are changed, the figure is built.
        :param normalize: if True the confusion matrix is normalized.
        :param colorscale: the color scale of the figure.
        """

        if not self.confusion_matrix_figure or not normalize or colorscale != 'Viridis':
            self.build_confusion_matrix_figure(normalize=normalize, colorscale=colorscale)

        iplot(self.confusion_matrix_figure)

    def build_roc_figure(self, title='ROC Curve'):
        """
        Building the ROC curve figure of the classifier.
        :param title: the Title of the plot.
        """
        data = list()

        if self.n_classes < 3:
            # False positive rate and true positive rate - computed from roc_curve()
            fpr, tpr, _ = roc_curve(self.dataset['target'], self.predicted_target_score[:, 1])

            # Area under curve.
            roc_auc = auc(fpr, tpr)

            # Updating the data list.
            data.append(go.Scatter(x=fpr,
                                   y=tpr,
                                   mode='lines',
                                   line=dict(color='darkorange'),
                                   name='ROC curve (area = %0.2f)' % roc_auc))
        else:

            # False Positive, True Positive rates and Area Under Curve values for each class.
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(self.n_classes):
                fpr[i], tpr[i], _ = roc_curve((self.dataset['target'] == i).astype(float),
                                              self.predicted_target_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

                data.append(go.Scatter(x=fpr[i],
                                       y=tpr[i],
                                       mode='lines',
                                       name='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i])))

        # Diagonal
        data.append(go.Scatter(x=[0, 1], y=[0, 1],
                               mode='lines',
                               line=dict(color='navy', dash='dash'),
                               showlegend=False))

        layout = go.Layout(title=title,
                           xaxis=dict(title='False Positive Rate'),
                           yaxis=dict(title='True Positive Rate'))

        self.roc_figure = go.Figure(data=data, layout=layout)

    def plot_roc(self, title='ROC Curve'):
        """
        Plotting the ROC curve figure with plotly's iplot function.
        :param title: plot title.
        """
        if not self.roc_figure:
            self.build_roc_figure(title=title)

        self.roc_figure['layout']['title'] = title

        iplot(self.roc_figure)

    def build_prediction_figure(self, title='Classifier Prediction'):
        """
        Building the classifier prediction figure.
        :param title: Title of the figure.
        """

        pass

    def plot_prediction(self, title='Classifier Prediction'):
        """
        Plotting the prediction figure with plotly's iplot function.
        :param title: plot title.
        """

        if not self.prediction_figure:
            self.build_prediction_figure(title=title)
        else:
            self.prediction_figure['layout']['title'] = title

        iplot(self.prediction_figure)

    def save_prediction_figure(self, file_name):
        """
        Saving the prediction figure as an html file.
        :param file_name: the html file name.
        """

        self.save_figure(self.prediction_figure, file_name)

    def save_roc_figure(self, file_name):
        """
        Saving the ROC curve figure as an html file.
        :param file_name: the html file name.
        """

        self.save_figure(self.roc_figure, file_name)

    def save_confusion_matrix_figure(self, file_name):
        """
        Saving the confusion matrix figure as an html file.
        :param file_name: the html file name.
        """

        self.save_figure(self.confusion_matrix_figure, file_name)


class TwoDimensionalClassifierPlotHandler(ClassifierPlotHandler):
    """ Handles all the plots related of the chosen classifier on 2D. """

    def __init__(self, dataset, trained_classifier, **params):
        """
        The initialization function of the 2D classifier plot handler.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_classifier: sklearn classifier (trained / fitted).
        :param params: other params.
        """

        dataset['data'] = dataset['data'][:, :2]

        super(TwoDimensionalClassifierPlotHandler, self).__init__(dataset, trained_classifier, **params)

    def build_prediction_figure(self, title='Classifier Prediction', step_size=0.01):
        """
        Building the classifier prediction figure.
        :param title: Title of the figure.
        :param step_size: Plot resolution.
        """

        data = list()

        x_min, x_max = self.dataset['data'][:, 0].min() - 1, self.dataset['data'][:, 0].max() + 1
        y_min, y_max = self.dataset['data'][:, 1].min() - 1, self.dataset['data'][:, 1].max() + 1

        x = np.arange(x_min, x_max, step_size)
        y = np.arange(y_min, y_max, step_size)
        x_mesh, y_mesh = np.meshgrid(x, y)

        z = self.trained_classifier.predict(np.column_stack((x_mesh.ravel(), y_mesh.ravel())))

        z = z.reshape(x_mesh.shape)

        data.append(go.Contour(x=x, y=y, z=z,
                               showscale=False,
                               colorscale='Viridis'))

        data.append(go.Scatter(x=self.dataset['data'][:, 0],
                               y=self.dataset['data'][:, 1],
                               mode='markers',
                               marker=dict(color=self.dataset['target'],
                                           showscale=False,
                                           colorscale='Reds',
                                           line=dict(color='black', width=1))))

        layout = go.Layout(title=title)

        self.prediction_figure = go.Figure(data=data, layout=layout)


class ThreeDimensionalClassifierPlotHandler(ClassifierPlotHandler):
    """ Handles all the plots related of the chosen classifier on 3D. """

    def __init__(self, dataset, trained_classifier, **params):
        """
        The initialization function of the 3D classifier plot handler.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_classifier: sklearn classifier (trained / fitted).
        :param params: other params.
        """

        dataset['data'] = dataset['data'][:, :3]

        super(ThreeDimensionalClassifierPlotHandler, self).__init__(dataset, trained_classifier, **params)

    def build_prediction_figure(self, title='Classifier Prediction'):
        """
        Plotting the classifier prediction and saving the figure.
        :param title: Title of the plot
        """

        labels = self.trained_classifier.predict(self.dataset['data'])

        data = [go.Scatter3d(x=self.dataset['data'][:, 0],
                             y=self.dataset['data'][:, 1],
                             z=self.dataset['data'][:, 2],
                             showlegend=False,
                             mode='markers',
                             marker=dict(
                                 color=labels.astype(np.float),
                                 line=dict(color='black', width=1)))]

        layout = go.Layout(title=title)

        self.prediction_figure = go.Figure(data=data, layout=layout)


import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot

from palantiri.BasePlotHandlers import PlotHandler


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
        self.confusion_matrix_colorscale = 'Viridis'

        self.prediction_figure = None
        self.roc_figure = None
        self.confusion_matrix_figure = None

        super(ClassifierPlotHandler, self).__init__(**params)

    @classmethod
    def from_pandas_dataframe(cls, dataframe, trained_classifier, **params):
        """
        Constructing the handler from a pandas dataframe.
        :param dataframe: the dataframe form which the handler is constructed.
        The 'target' column  should be included in the dataframe.
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

    def build_confusion_matrix(self, normalize=False):
        """
        Building the confusion matrix
        :param normalize: if True confusion matrix is normalized.
        """

        prediction = self.trained_classifier.predict(self._dataset['data'])

        self._confusion_matrix = confusion_matrix(self._dataset['target'], prediction)

        if normalize:
            self._confusion_matrix = \
                self._confusion_matrix.astype('float') / self._confusion_matrix.sum(axis=1)[:, np.newaxis]
        else:
            self._confusion_matrix = self._confusion_matrix

    def build_confusion_matrix_figure(self, figure_layout):
        """
        Builds the confusion matrix figure in confusion_matrix_figure.
        :param figure_layout: figure layout - plot.ly layout object.
        """

        if not self._confusion_matrix:
            self.build_confusion_matrix()

        cm = np.flipud(self._confusion_matrix)
        x = list(self.class_names)
        y = list(reversed(self.class_names))

        self.confusion_matrix_figure = ff.create_annotated_heatmap(z=cm, x=x, y=y,
                                                                   colorscale=self.confusion_matrix_colorscale)

        self.confusion_matrix_figure['layout'].update(figure_layout)

    def plot_confusion_matrix(self, figure_layout=None):
        """
        Plotting the confusion matrix figure with plot.ly's iplot function.
        :param figure_layout: figure layout - plot.ly layout object.
        """

        if not figure_layout:
            figure_layout = go.Layout(
                xaxis={'title': 'Confusion Matrix <br /><br />Predicted Value'},
                yaxis={'title': 'True Value'})

        if not self.confusion_matrix_figure:
            self.build_confusion_matrix_figure(figure_layout)
        else:
            self.confusion_matrix_figure['layout'].update(figure_layout)

        iplot(self.confusion_matrix_figure)

    def build_roc_figure(self, figure_layout=go.Layout()):
        """
        Building the ROC curve figure of the classifier.
        :param figure_layout: figure layout - plot.ly layout object.
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
                                   hoverinfo='y',
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
                                       hoverinfo='y',
                                       mode='lines',
                                       name='ROC curve of class {0} (area = {1:0.2f})'''.format(
                                           self.class_names[i], roc_auc[i])))

        # Diagonal
        data.append(go.Scatter(x=[0, 1], y=[0, 1],
                               mode='lines',
                               hoverinfo='skip',
                               line=dict(color='navy', dash='dash'),
                               showlegend=False))

        self.roc_figure = go.Figure(data=data, layout=figure_layout)

    def plot_roc(self, figure_layout=None):
        """
        Plotting the ROC curve figure with plot.ly's iplot function.
        :param figure_layout: figure layout - plot.ly Layout object.
        """

        if not figure_layout:
            figure_layout = go.Layout(title=dict(text='ROC Curve', x=0.5),
                                      xaxis=dict(title='False Positive Rate'),
                                      yaxis=dict(title='True Positive Rate'))

        if not self.roc_figure:
            self.build_roc_figure(figure_layout=figure_layout)
        else:
            self.roc_figure['layout'].update(figure_layout)

        iplot(self.roc_figure)

    def build_prediction_figure(self, figure_layout):
        """
        Building the classifier prediction figure.
        :param figure_layout: figure layout - plot.ly Layout object.
        """

        pass

    def plot_prediction(self, figure_layout=None):
        """
        Plotting the prediction figure with plot.ly's iplot function.
        :param figure_layout: figure layout - plot.ly Layout object.
        """

        if not figure_layout:
            figure_layout = go.Layout(title=dict(text='Classifier Prediction', x=0.5))

        if not self.prediction_figure:
            self.build_prediction_figure(figure_layout=figure_layout)
        else:
            self.prediction_figure['layout'].update(figure_layout)

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

    def build_prediction_figure(self, figure_layout=go.Layout(), step_size=0.01):
        """
        Building the classifier prediction figure.
        :param figure_layout: figure layout - plot.ly Layout object.
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
                               hoverinfo='skip',
                               colorscale='Viridis'))

        data.append(go.Scatter(x=self.dataset['data'][:, 0],
                               y=self.dataset['data'][:, 1],
                               text=[self.class_names[i] for i in self.dataset['target']],
                               hoverinfo='text',
                               mode='markers',
                               marker=dict(color=self.dataset['target'],
                                           showscale=False,
                                           colorscale='Reds',
                                           line=dict(color='black', width=1))))

        if 'feature_names' in self.dataset.keys():
            figure_layout['xaxis'].update({'title': self.dataset['feature_names'][0]})
            figure_layout['yaxis'].update({'title': self.dataset['feature_names'][1]})

        self.prediction_figure = go.Figure(data=data, layout=figure_layout)


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

    def build_prediction_figure(self, figure_layout=go.Layout()):
        """
        Plotting the classifier prediction and saving the figure.
        :param figure_layout: figure layout - plot.ly Layout object.
        """

        labels = self.trained_classifier.predict(self.dataset['data'])

        data = list()

        for label in set(labels):

            data_points = self.dataset['data'][np.in1d(labels, np.asarray(label))]

            data.append(go.Scatter3d(x=data_points[:, 0],
                                     y=data_points[:, 1],
                                     z=data_points[:, 2],
                                     text=self.class_names[label],
                                     hoverinfo='text',
                                     showlegend=True,
                                     name=self.class_names[label],
                                     mode='markers',
                                     marker=dict(
                                         line=dict(color='black', width=1))))

        if 'feature_names' in self.dataset.keys():
            figure_layout['scene'].update(
                dict(xaxis={'title': self.dataset['feature_names'][0]},
                     yaxis={'title': self.dataset['feature_names'][1]},
                     zaxis={'title': self.dataset['feature_names'][2]}))

        self.prediction_figure = go.Figure(data=data, layout=figure_layout)

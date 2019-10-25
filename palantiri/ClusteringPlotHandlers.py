import numpy as np

import plotly.graph_objs as go
from plotly.offline import iplot

from palantiri.BasePlotHandlers import PlotHandler


class ClusteringPlotHandler(PlotHandler):
    """ Handles all the plots related of the chosen cluster. """

    def __init__(self, dataset, trained_cluster, **params):
        """
        Initialization function
        :param dataset: the dataset in a dict format with the following keys:
                        'data' - numpy array with all the data points.
        :param trained_cluster: sklearn cluster (trained / fitted).
        :param params: other params
        """

        self._dataset = dataset
        self._trained_cluster = trained_cluster

        self.prediction_figure = None

        if hasattr(self.trained_cluster, 'n_clusters'):
            self.n_clusters = self.trained_cluster.n_clusters
        elif hasattr(self.trained_cluster, 'labels_'):
            self.n_clusters = len([i for i in set(self.trained_cluster.labels_) if i >= 0])
        else:
            raise Exception('Number of clusters is not defined.')

        self.classes_names = ['Class {0}'.format(i) for i in range(self.n_clusters)]

        super(ClusteringPlotHandler, self).__init__(**params)

    @classmethod
    def from_pandas_dataframe(cls, dataframe, trained_cluster, **params):
        """
        Constructing the handler from a pandas dataframe.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_cluster: sklearn cluster (trained / fitted).
        :param params: other params.
        :return: returns the classifier plot handler object.
        """

        dataset = dict()
        dataset['data'] = dataframe.drop('target', axis=1).values
        dataset['feature_names'] = dataframe.drop('target', axis=1).columns.values
        return cls(dataset, trained_cluster, **params)

    @property
    def trained_cluster(self):
        """
        The trained cluster.
        :return: The cluster in the sklearn format.
        """
        return self._trained_cluster

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
            figure_layout = go.Layout(title=dict(text='Cluster Prediction', x=0.5))

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


class TwoDimensionalClusteringPlotHandler(ClusteringPlotHandler):
    """ Handles all the plots related of the chosen cluster on 2D. """

    def __init__(self, dataset, trained_cluster, **params):
        """
        The initialization function of the 2D classifier plot handler.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_cluster: sklearn cluster (trained / fitted).
        :param params: other params.
        """

        dataset['data'] = dataset['data'][:, :2]

        super(TwoDimensionalClusteringPlotHandler, self).__init__(dataset, trained_cluster, **params)

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

        z = self.trained_cluster.predict(np.column_stack((x_mesh.ravel(), y_mesh.ravel())))

        z = z.reshape(x_mesh.shape)

        data.append(go.Contour(x=x, y=y, z=z,
                               showscale=False,
                               hoverinfo='skip',
                               colorscale='Viridis'))

        data.append(go.Scatter(x=self.dataset['data'][:, 0],
                               y=self.dataset['data'][:, 1],
                               text=[self.classes_names[i] for i in self.trained_cluster.predict(self.dataset['data'])],
                               hoverinfo='text',
                               mode='markers',
                               marker=dict(color=self.trained_cluster.predict(self.dataset['data']),
                                           showscale=False,
                                           colorscale='Reds',
                                           line=dict(color='black', width=1))))

        if 'feature_names' in self.dataset.keys():
            figure_layout['xaxis'].update({'title': self.dataset['feature_names'][0]})
            figure_layout['yaxis'].update({'title': self.dataset['feature_names'][1]})

        self.prediction_figure = go.Figure(data=data, layout=figure_layout)


class ThreeDimensionalClusteringPlotHandler(ClusteringPlotHandler):
    """ Handles all the plots related of the chosen cluster on 3D. """

    def __init__(self, dataset, trained_cluster, **params):
        """
        The initialization function of the 3D classifier plot handler.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_cluster: sklearn cluster (trained / fitted).
        :param params: other params.
        """

        dataset['data'] = dataset['data'][:, :3]

        super(ThreeDimensionalClusteringPlotHandler, self).__init__(dataset, trained_cluster, **params)

    def build_prediction_figure(self, figure_layout=go.Layout()):
        """
        Plotting the cluster prediction and saving the figure.
        :param figure_layout: figure layout - plot.ly Layout object.
        """

        labels = self.trained_cluster.predict(self.dataset['data'])

        data = list()

        for label in set(labels):

            data_points = self.dataset['data'][np.in1d(labels, np.asarray(label))]
            data.append(go.Scatter3d(x=data_points[:, 0],
                                     y=data_points[:, 1],
                                     z=data_points[:, 2],
                                     text=self.classes_names[label],
                                     hoverinfo='text',
                                     showlegend=True,
                                     name=self.classes_names[label],
                                     mode='markers',
                                     marker=dict(
                                         line=dict(color='black', width=1))))

        if 'feature_names' in self.dataset.keys():
            figure_layout['scene'].update(
                dict(xaxis={'title': self.dataset['feature_names'][0]},
                     yaxis={'title': self.dataset['feature_names'][1]},
                     zaxis={'title': self.dataset['feature_names'][2]}))

        self.prediction_figure = go.Figure(data=data, layout=figure_layout)

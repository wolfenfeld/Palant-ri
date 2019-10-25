import numpy as np

import plotly.graph_objs as go
from plotly.offline import iplot

from palantiri.BasePlotHandlers import PlotHandler


class RegressionPlotHandler(PlotHandler):
    """ Handles all the plots related of the chosen regressor. """

    def __init__(self, dataset, trained_regressor, **params):
        """
        Initialization function
        :param dataset: the dataset in a dict format with the following keys:
                        'data' - numpy array with all the data points.
                        'target' - the label of the corresponding data point.
                        'target_names' - the target name.

        :param trained_regressor: sklearn regressor(trained / fitted)..
        :param params: other params
        """

        self._dataset = dataset
        self._trained_regressor = trained_regressor

        self.prediction_figure = None

        self.target_name = 'Target'

        super(RegressionPlotHandler, self).__init__(**params)

    @classmethod
    def from_pandas_dataframe(cls, dataframe, trained_regressor, **params):
        """
        Constructing the handler from a pandas dataframe.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_regressor: sklearn regressor (trained / fitted).
        :param params: other params.
        :return: returns the classifier plot handler object.
        """

        assert 'target' in dataframe.columns.values, 'target values not in dataframe'

        dataset = dict()
        dataset['data'] = dataframe.drop('target', axis=1).values
        dataset['target'] = dataframe['target'].values
        dataset['feature_names'] = dataframe.drop('target', axis=1).columns.values
        return cls(dataset, trained_regressor, **params)

    @property
    def trained_regressor(self):
        """
        The trained regressor.
        :return: The regressor in sklearn format.
        """
        return self._trained_regressor

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
        Building the regression figure.
        :param figure_layout: figure layout - plot.ly layout object.
        """

        pass

    def plot_prediction(self, figure_layout=None):
        """
        Plotting the regression figure with plot.ly's iplot function.
        :param figure_layout: figure layout - plot.ly layout object.
        """

        if not figure_layout:
            figure_layout = go.Layout(title=dict(text='Regression Plot', x=0.5))

        if not self.prediction_figure:
            self.build_prediction_figure(figure_layout)
        else:
            self.prediction_figure['layout'].update(figure_layout)

        iplot(self.prediction_figure)

    def save_prediction_figure(self, file_name):
        """
        Saving the prediction figure as an html file.
        :param file_name: the html file name.
        """

        self.save_figure(self.prediction_figure, file_name)


class OneDimensionalRegressionPlotHandler(RegressionPlotHandler):
    """ Handles all the plots related of the chosen 1D regression. """

    def __init__(self, dataset, trained_regressor, **params):
        """
        The initialization function of the 2D classifier plot handler.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_regressor: sklearn regressor (trained / fitted).
        :param params: other params.
        """
        dataset['data'] = dataset['data'][:, :1]

        super(OneDimensionalRegressionPlotHandler, self).__init__(dataset, trained_regressor, **params)

    def build_prediction_figure(self, figure_layout, step_size=0.1, x_range=None):
        """
        Building the regression figure.
        :param figure_layout: figure layout - plot.ly layout object.
        :param step_size: resolution of the x-axis.
        :param x_range: the range of the prediction (x-axis),
        list of 2 numbers - indicating the start and end of the range
        if none will take the minimum and maximum of the data set.
        """

        if not x_range:
            x = np.arange(min(self.dataset['data']), max(self.dataset['data']), step_size).reshape(-1, 1)
        else:
            x = np.arange(min(x_range), max(x_range), step_size).reshape(-1, 1)

        data = [go.Scatter(x=self.dataset['data'][:, 0],
                           y=self.dataset['target'],
                           showlegend=False,
                           hoverinfo='skip',
                           mode='markers',
                           marker=dict(
                               line=dict(color='black', width=1))),
                go.Scatter(x=x.ravel(),
                           y=self.trained_regressor.predict(x).ravel(),
                           hoverinfo='y',
                           showlegend=False,
                           mode='lines')]

        if 'feature_names' in self.dataset.keys():
            figure_layout['xaxis'].update({'title': self.dataset['feature_names'][0]})
            figure_layout['yaxis'].update({'title': self.target_name})

        self.prediction_figure = go.Figure(data=data, layout=figure_layout)


class TwoDimensionalRegressionPlotHandler(RegressionPlotHandler):
    """ Handles all the plots related of the chosen regressor on 2D. """

    def __init__(self, dataset, trained_regressor, **params):
        """
        The initialization function of the 3D classifier plot handler.
        :param dataframe: the dataframe form which the handler is constructed.
        :param trained_regressor: sklearn regressor(trained / fitted).
        :param params: other params.
        """

        dataset['data'] = dataset['data'][:, :2]

        super(TwoDimensionalRegressionPlotHandler, self).__init__(dataset, trained_regressor, **params)

    def build_prediction_figure(self,  figure_layout=go.Layout(), x_range=None, y_range=None, step_size=0.1):
        """
        Building the regression figure.
        :param figure_layout: figure layout - plot.ly layout object.
        :param step_size: resolution of the x-axis.
        :param x_range: the range of the prediction (x-axis),
        list of 2 numbers - indicating the start and end of the range
        if none will take the minimum and maximum of the data set.
        :param y_range: similar to x_range for the y-axis.
        """

        if not x_range:
            x = np.arange(min(self.dataset['data'][:, 0]), max(self.dataset['data'][:, 0]), step_size)
        else:
            x = np.arange(min(x_range), max(x_range), step_size)

        if not y_range:
            y = np.arange(min(self.dataset['data'][:, 1]), max(self.dataset['data'][:, 1]), step_size)
        else:
            y = np.arange(min(y_range), max(y_range), step_size)

        x_mesh, y_mesh = np.meshgrid(x, y)

        z = self.trained_regressor.predict(np.column_stack((x_mesh.ravel(), y_mesh.ravel()))).reshape(x_mesh.shape)

        data = [go.Surface(x=x, y=y, z=z,
                           showscale=False,
                           colorscale='Viridis',
                           hoverinfo='z'),
                go.Scatter3d(x=self.dataset['data'][:, 0],
                             y=self.dataset['data'][:, 1],
                             z=self.dataset['target'],
                             hoverinfo='skip',
                             mode='markers',
                             marker=dict(showscale=False,
                                         colorscale='Reds',
                                         line=dict(color='black', width=0.3)))]

        if 'feature_names' in self.dataset.keys():
            figure_layout['scene'].update(
                dict(xaxis={'title': self.dataset['feature_names'][0]},
                     yaxis={'title': self.dataset['feature_names'][1]},
                     zaxis={'title': self.target_name}))

        self.prediction_figure = go.Figure(data=data, layout=figure_layout)

import unittest

from plotly import graph_objs as go

import networkx as nx

from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.datasets import load_iris, load_breast_cancer, load_boston

from palantiri.ClassificationPlotHandlers import TwoDimensionalClassifierPlotHandler
from palantiri.ClassificationPlotHandlers import ThreeDimensionalClassifierPlotHandler
from palantiri.GraphPlotHandlers import GraphPlotHandler
from palantiri import OneDimensionalRegressionPlotHandler
from palantiri import TwoDimensionalRegressionPlotHandler
from palantiri import TwoDimensionalClusteringPlotHandler
from palantiri import ThreeDimensionalClusteringPlotHandler


class ClassifierPlotHandlersTests(unittest.TestCase):

    def test_classifiers_two_dimensional_plot_handler(self):
        iris = load_iris()

        iris_clf = svm.SVC(kernel='rbf', probability=True, gamma='auto')
        iris_clf.fit(iris.data[:, :2], iris.target)

        plot_handler = TwoDimensionalClassifierPlotHandler(iris, iris_clf)

        plot_handler.build_confusion_matrix_figure(figure_layout=go.Layout(title='test'))
        plot_handler.build_prediction_figure(figure_layout=go.Layout(title='test'))
        plot_handler.build_roc_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.confusion_matrix_figure.data[0].type, 'heatmap')
        self.assertEqual(plot_handler.prediction_figure.data[0].type, 'contour')
        self.assertEqual(plot_handler.roc_figure.data[0].type, 'scatter')

    def test_classifiers_three_dimensional_plot_handler(self):

        breast_cancer = load_breast_cancer()
        breast_cancer_clf = svm.SVC(kernel='rbf', probability=True, gamma='auto')
        breast_cancer_clf.fit(breast_cancer.data[:, :3], breast_cancer.target)

        plot_handler = ThreeDimensionalClassifierPlotHandler(breast_cancer, breast_cancer_clf)

        plot_handler.build_confusion_matrix_figure(figure_layout=go.Layout(title='test'))
        plot_handler.build_prediction_figure(figure_layout=go.Layout(title='test'))
        plot_handler.build_roc_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.confusion_matrix_figure.data[0].type, 'heatmap')
        self.assertEqual(plot_handler.prediction_figure.data[0].type, 'scatter3d')
        self.assertEqual(plot_handler.roc_figure.data[0].type, 'scatter')


class RegressionPlotHandlersTests(unittest.TestCase):

    def test_regressor_one_dimensional_plot_handler(self):

        boston = load_boston()
        boston['data'] = boston['data'][:, 5].reshape(-1, 1)
        boston['feature_names'] = boston['feature_names'][5]

        boston_regressor = RandomForestRegressor(n_estimators=10)
        boston_regressor.fit(boston['data'], boston['target'])

        plot_handler = OneDimensionalRegressionPlotHandler(boston, boston_regressor)

        plot_handler.build_prediction_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.prediction_figure.data[0].type, 'scatter')

    def test_classifiers_tow_dimensional_plot_handler(self):

        boston = load_boston()
        boston['data'] = boston['data'][:, (2, 5)].reshape(-1, 2)
        boston['feature_names'] = boston['feature_names'][2], boston['feature_names'][5]

        boston_regressor = RandomForestRegressor(n_estimators=10)
        boston_regressor.fit(boston['data'], boston['target'])

        plot_handler = TwoDimensionalRegressionPlotHandler(boston, boston_regressor)

        plot_handler.build_prediction_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.prediction_figure.data[0].type, 'surface')


class ClusteringPlotHandlersTests(unittest.TestCase):

    def test_clustering_two_dimensional_plot_handler(self):

        iris = load_iris()

        iris_cluster = AffinityPropagation()
        iris_cluster.fit(iris.data[:, :2])

        plot_handler = TwoDimensionalClusteringPlotHandler(iris, iris_cluster)

        plot_handler.build_prediction_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.prediction_figure.data[0].type, 'contour')

    def test_clustering_three_dimensional_plot_handler(self):
        iris = load_iris()

        iris_cluster = KMeans(n_clusters=3)
        iris_cluster.fit(iris.data[:, :3])

        plot_handler = ThreeDimensionalClusteringPlotHandler(iris, iris_cluster)

        plot_handler.build_prediction_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.prediction_figure.data[0].type, 'scatter3d')


class GraphPlotHandlersTests(unittest.TestCase):

    def test_graph_plot_handler(self):

        karate_club_graph = nx.karate_club_graph()

        plot_handler = GraphPlotHandler(karate_club_graph, node_data_key='club')
        plot_handler.build_graph_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.graph_figure.data[0].type, 'scatter')

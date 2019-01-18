import unittest

from plotly import graph_objs as go

from sklearn import svm
from sklearn.datasets import load_iris, load_breast_cancer

from palantiri.ClassificationPlotHandlers import TwoDimensionalClassifierPlotHandler
from palantiri.ClassificationPlotHandlers import ThreeDimensionalClassifierPlotHandler

iris = load_iris()

iris_clf = svm.SVC(kernel='rbf', probability=True, gamma='auto')
iris_clf.fit(iris.data[:, :2], iris.target)

breast_cancer = load_breast_cancer()

breast_cancer_clf = svm.SVC(kernel='rbf',probability=True,gamma='auto')
breast_cancer_clf.fit(breast_cancer.data[:, :3], breast_cancer.target)


class PlotHandlersTests(unittest.TestCase):

    def test_classifiers_two_dimensional_plot_handler(self):

        plot_handler = TwoDimensionalClassifierPlotHandler(iris, iris_clf)

        plot_handler.build_confusion_matrix_figure(figure_layout=go.Layout(title='test'))
        plot_handler.build_prediction_figure(figure_layout=go.Layout(title='test'))
        plot_handler.build_roc_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.confusion_matrix_figure.data[0].type, 'heatmap')
        self.assertEqual(plot_handler.prediction_figure.data[0].type, 'contour')
        self.assertEqual(plot_handler.roc_figure.data[0].type, 'scatter')

    def test_classifiers_tree_dimensional_plot_handler(self):

        plot_handler = ThreeDimensionalClassifierPlotHandler(breast_cancer, breast_cancer_clf)

        plot_handler.build_confusion_matrix_figure(figure_layout=go.Layout(title='test'))
        plot_handler.build_prediction_figure(figure_layout=go.Layout(title='test'))
        plot_handler.build_roc_figure(figure_layout=go.Layout(title='test'))

        self.assertEqual(plot_handler.confusion_matrix_figure.data[0].type, 'heatmap')
        self.assertEqual(plot_handler.prediction_figure.data[0].type, 'scatter3d')
        self.assertEqual(plot_handler.roc_figure.data[0].type, 'scatter')

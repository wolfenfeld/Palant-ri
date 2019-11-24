---
layout: page

title: Classifier Visualisations

path: /get_started/classifiers

categories: [get_started] 
---


# Classifiers
In this tutorial we will show how to use Palantíri with Scikit-Learn classifiers.

We will start with importing the relevant classifiers and data:
```python
from sklearn import svm
from sklearn.datasets import load_iris, load_breast_cancer
```                                              

Next we will load the data and train the classifier:

```python 
iris = load_iris()

iris_clf = svm.SVC(kernel='rbf',probability=True,gamma='auto')
iris_clf.fit(iris.data[:,:2], iris.target);
```

We will start with a *Two Dimensional Plot Handler* for the classifier.

The plot handler is initialized with the trained classifier and data points that will be displayed.

*Important note - for a two dimensional plot handler that classifier must be trained on two features.*
```python
from palantiri.ClassificationPlotHandlers import TwoDimensionalClassifierPlotHandler

plot_handler = TwoDimensionalClassifierPlotHandler(iris, iris_clf)
```

Now that everything is set-up we can plot the classification along with the data:

```python
plot_handler.plot_prediction()
```
{% include scripts/plots/2d-prediction-plot.html %}
```python
plot_handler.plot_roc()
```
{% include scripts/plots/2d-prediction-roc.html %}
```python
plot_handler.plot_confusion_matrix()
```

{% include scripts/plots/2d-prediction-confusion-matrix.html %}

In the previous example we have used a classifier that we trained on two features.

We will now show how to use the *Three Dimensional Plot Handler*

We will start with loading the data and training the classifier: 
```python
breast_cancer = load_breast_cancer()

breast_cancer_clf = svm.SVC(kernel='rbf',probability=True,gamma='auto')
breast_cancer_clf.fit(breast_cancer.data[:, :3],breast_cancer.target);
```

We will initialize that plot handler.

*Important note - for a three dimensional plot handler that classifier must be trained on three features.*

```python
from palantiri.ClassificationPlotHandlers import ThreeDimensionalClassifierPlotHandler

plot_handler = ThreeDimensionalClassifierPlotHandler(breast_cancer, breast_cancer_clf)
```
Not like in the *Two Dimensional Plot Handler*, the data points here will be classified with the trained model.


Once everything is set up we can start plotting:
                      
```python
plot_handler.plot_prediction()

```
{% include scripts/plots/3d-prediction-plot.html %}
```python
plot_handler.plot_roc()

```
{% include scripts/plots/3d-prediction-roc.html %}
```python
plot_handler.plot_confusion_matrix()
```
{% include scripts/plots/3d-prediction-confusion-matrix.html %}

To conclude - we showed how simple it is to take a trained classier and data points and plot them with the Palantíri package.
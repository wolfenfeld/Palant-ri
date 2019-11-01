---
layout: post
title: Classifier Visualisations
categories: [get_started] 
---
# Classifiers

```python

from sklearn import svm
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
```

```python 
breast_cancer = load_breast_cancer()

breast_cancer_clf = svm.SVC(kernel='rbf',probability=True,gamma='auto')
breast_cancer_clf.fit(breast_cancer.data[:, :3],breast_cancer.target);
```

```python
from palantiri.ClassificationPlotHandlers import TwoDimensionalClassifierPlotHandler

plot_handler = TwoDimensionalClassifierPlotHandler(iris, iris_clf)
```

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

{% include scripts/plots/3d-prediction-plot.html %}
{% include scripts/plots/3d-prediction-roc.html %}
{% include scripts/plots/3d-prediction-confusion-matrix.html %}
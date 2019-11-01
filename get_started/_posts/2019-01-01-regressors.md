---
layout: post
title: Regressors Visualisations
categories: [get_started] 
---
# Regressors
In this tutorial we will show how to use Palantíri with Scikit-Learn regressors.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
```

```python
dataset = load_boston()
dataset['data'] = dataset['data'][:,5].reshape(-1,1)
dataset['feature_names'] = dataset['feature_names'][5]
```

```python
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(dataset['data'],dataset['target'])
```

```python
from palantiri.RegressionPlotHandlers import OneDimensionalRegressionPlotHandler
plot_handler = OneDimensionalRegressionPlotHandler(dataset=dataset, trained_regressor=regressor)
```

```python
plot_handler.plot_prediction()
```
{% include scripts/plots/regression-plot.html %}

```python
dataset = load_boston()
dataset['data'] = dataset['data'][:,(2,5)].reshape(-1,2)
dataset['feature_names'] = dataset['feature_names'][2],dataset['feature_names'][5]
```

```python
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(dataset['data'],dataset['target'])
```

```python
from palantiri.RegressionPlotHandlers import TwoDimensionalRegressionPlotHandler
plot_handler = TwoDimensionalRegressionPlotHandler(dataset=dataset,trained_regressor=regressor)
```

```python
plot_handler.build_prediction_figure(step_size=2)
plot_handler.plot_prediction()
```
{% include scripts/plots/3d-regression-plot.html %}

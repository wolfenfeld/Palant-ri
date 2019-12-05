---
layout: page
title: Regressors Visualisations
path: /get_started/regressors

categories: [get_started] 
---
In this tutorial we will show how to use Palantíri with Scikit-Learn regressors.

# One Dimensional Regression Model

We will start with a *One Dimensional Regression Plot Handler*.

We first import a regressor and some data.
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
```
We will now load the data:
```python
dataset = load_boston()
dataset['data'] = dataset['data'][:,5].reshape(-1,1)
dataset['feature_names'] = dataset['feature_names'][5]
```
Train the regressor:
```python
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(dataset['data'],dataset['target'])
```
And initialize the plot handler with the trained regressor and tha data points that we wish to display:
```python
from palantiri.RegressionPlotHandlers import OneDimensionalRegressionPlotHandler
plot_handler = OneDimensionalRegressionPlotHandler(dataset=dataset, trained_regressor=regressor)
```
Once everything is ready we can start with the visualizations:

```python
plot_handler.plot_prediction()
```
{% include scripts/plots/regression-plot.html %}

# Two Dimensional Regression Model

Next we will use the *Two Dimensional Plot Handler*:

We will load the data:
```python
dataset = load_boston()
dataset['data'] = dataset['data'][:,(2,5)].reshape(-1,2)
dataset['feature_names'] = dataset['feature_names'][2],dataset['feature_names'][5]
```
Train the model:
```python
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(dataset['data'],dataset['target'])
```
And initialize the regressor with the trained regressor and the data points that we wish to display:
```python
from palantiri.RegressionPlotHandlers import TwoDimensionalRegressionPlotHandler
plot_handler = TwoDimensionalRegressionPlotHandler(dataset=dataset,trained_regressor=regressor)
```
Once verything is set we need can continue with the visualizations:
```python
plot_handler.build_prediction_figure(step_size=2)
plot_handler.plot_prediction()
```
{% include scripts/plots/3d-regression-plot.html %}

To conclude - once we have a trained regressor and data sample that we wish to display it is super quick and easy to display the model.
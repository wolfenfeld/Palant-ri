---
layout: page
title: Regret Visualisations
path: /get_started/regret

categories: [get_started] 
---

# Regret

When dealing with *Reinforcement Learning* algorithms, one way to measure performance of the algorithm is with *Mean Accumulated Regret*.

The *Mean Accumulated Regret* is defined as follows:

$$ R(t) = E[r(t)] $$

where $$r(t)$$ is defined as the difference between the reward at time *t* and the optimal reward that can be achieved at time *t*.

For more information please see this blog [post](https://jewpyter.com/machine-learning/2019-03-01-MAB-post/)

In this tutorial we will show how to visualize the *Mean Accumulated Regret*.

We will start with generating the data:
```python 
data = {'Algorithm 1':np.array([np.log(np.linspace(1,1000,1000))+ 5*np.random.rand(1000) for i in range(30)]).T,
        'Algorithm 2':np.array([3*np.log(np.linspace(1,1000,1000))+ 4*np.random.rand(1000) for i in range(30)]).T}
```

In this example we generate data for two different algorithms. 

The data for each algorithm, the accumulated regret for round *t* in run *n*, is a T by N numpy array, where *T* (1000 in this case)  is the number of rounds in each "run" and *N* is the total number of runs (30 in our case).

We will use the *RegretPlotHandler* to visualize the *Mean Accumulated Regret*:

```python
from palantiri.RegretPlotHandler import RegretPlotHandler
plot_handler = RegretPlotHandler(cumulative_regret_data=data)
```
Once hte plot handler is initialized we can plot:
```python 
plot_handler.plot_regret()
```

{% include scripts/plots/regret_plot.html %}

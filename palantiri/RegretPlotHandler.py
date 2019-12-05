from plotly import graph_objs as go
from plotly.offline import iplot
import colorlover as cl

from palantiri.BasePlotHandlers import PlotHandler


class RegretPlotHandler(PlotHandler):
    """
    The regret plot handler - handles all the regret related plots.
    """

    def __init__(self, cumulative_regret_data, **params):
        """
        Initialization function
        :param cumulative_regret_data: a dictionary where the keys are the dataset name and the values
               are T by N numpy arrays, where T is the horizon and N is the number of runs.
        :param params: other params
        """
        self._cumulative_regret_data = cumulative_regret_data
        self.regret_figure = None
        self.colors = cl.scales['7']['qual']['Accent']

        super(RegretPlotHandler, self).__init__(**params)

    def build_graph_figure(self, figure_layout=go.Layout()):
        """
        Building the regret plot figure.
        :param figure_layout: a plot.ly layout object.
        """
        assert len(self._cumulative_regret_data.keys()) <= 7, 'More Colors are needed, update the colors attribute.'
        data = list()
        colors = self.colors.copy()
        for dataset_name, dataset in self._cumulative_regret_data.items():
            color = colors.pop()
            mean = dataset.mean(axis=1)
            std = dataset.std(axis=1)

            x = list(range(len(mean)))
            x_reversed = x[::-1]
            y = mean.tolist()
            y_upper = (mean + std).tolist()
            y_lower = (mean - std).tolist()
            y_lower = y_lower[::-1]

            data.append(go.Scatter(
                x=x + x_reversed,
                y=y_upper + y_lower,
                fill='tozerox',
                fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.2)'),
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=dataset_name
            ))
            data.append(go.Scatter(
                x=x,
                y=y,
                line=dict(color=color),
                mode='lines',
                name=dataset_name,
            ))

        self.regret_figure = go.Figure(data=data, layout=figure_layout)

    def plot_regret(self, figure_layout=None):
        """
        Plotting the graph figure.
        :param figure_layout: a plot.ly layout object.
        """

        if not figure_layout:
            figure_layout = go.Layout(title=dict(text='Regret Plot', x=0.5),
                                      xaxis=dict(
                                          title='Round',
                                          autorange=True,
                                          showgrid=False,
                                          zeroline=False,
                                          showline=False,
                                          ticks='',
                                          showticklabels=True),
                                      yaxis=dict(
                                          title='Mean Accumulated Regret',
                                          autorange=True,
                                          showgrid=False,
                                          zeroline=False,
                                          showline=False,
                                          ticks='',
                                          showticklabels=True))

        if not self.regret_figure:
            self.build_graph_figure(figure_layout=figure_layout)

        iplot(self.regret_figure)

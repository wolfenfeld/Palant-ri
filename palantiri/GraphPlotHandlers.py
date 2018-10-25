from palantiri.BasePlotHandlers import PlotHandler

import networkx as nx

import plotly.graph_objs as go
from plotly.offline import iplot


class GraphPlotHandler(PlotHandler):
    """
    The graph plot handler - handles all the graph related plots.
    """
    def __init__(self, graph, node_data_key=None, **params):
        """
        Initialization function
        :param graph: the graph to be plotted.
        :param node_data_key: the location of the relevant data within each node.
        :param params: other params
        """
        self._graph = graph
        self.node_data_key = node_data_key
        self.graph_figure = None

        super(GraphPlotHandler, self).__init__(**params)

    @property
    def graph(self):
        """
        Getter for the graph
        :return: the graph
        """
        return self._graph

    @graph.setter
    def graph(self, graph):
        """
        Setter for the graph
        :param graph: the new graph.
        """
        self._graph = graph

    def _get_edge_trace(self, pos):
        """
        Constructing the edge trace.
        :param pos: the position of each node : {node_number:[x,y]}
        :return: a plot.ly trace object of the edges.
        """
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False)

        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        return edge_trace

    def _get_node_trace(self, pos):
        """
        Constructing the node trace
        :param pos: the position of each node: {node_number:[x,y]}.
        :return: a plot.ly trace object og the nodes
        """

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            showlegend=False,
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=17,
                line=dict(width=2)))

        for node, node_data in self.graph.nodes(data=True):
            x, y = pos[node]

            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node_data[self.node_data_key]])
            node_trace['marker']['color'] += tuple([hash(node_data[self.node_data_key]) % 256])
        return node_trace

    def build_graph_figure(self, figure_layout=go.Layout(), pos=None):
        """
        Building the graph plot figure.
        :param figure_layout: a plot.ly layout object.
        :param pos: the position of each node - if none: spring layout is used for position - {node_number:[x,y]}.
        """

        if not pos:
            pos = nx.spring_layout(self._graph)

        data = [self._get_edge_trace(pos), self._get_node_trace(pos)]

        self.graph_figure = go.Figure(data=data, layout=figure_layout)

    def plot_graph(self, figure_layout=None):
        """
        Plotting the graph figure.
        :param figure_layout: a plot.ly layout object.
        """

        if not figure_layout:
            figure_layout = go.Layout(title='Graph Plot',
                                      xaxis=dict(
                                          autorange=True,
                                          showgrid=False,
                                          zeroline=False,
                                          showline=False,
                                          ticks='',
                                          showticklabels=False),
                                      yaxis=dict(
                                          autorange=True,
                                          showgrid=False,
                                          zeroline=False,
                                          showline=False,
                                          ticks='',
                                          showticklabels=False))

        if not self.graph_figure:
            self.build_graph_figure(figure_layout=figure_layout)

        iplot(self.graph_figure)

from io import StringIO
import networkx as nx
import pydotplus
from sklearn.tree import export_graphviz

from palantiri.BasePlotHandlers import PlotHandler

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

    @classmethod
    def from_decision_tree(cls, decision_tree, feature_names, **params):
        """
        Handler from Decision tree
        :param decision_tree:  sklearn decision tree
        :param feature_names: the feature names
        :param params: other params
        :return: graph plot handler for the decision tree.
        """
        dot_data = StringIO()

        class_names = [str(name) for name in decision_tree.classes_]

        export_graphviz(decision_tree, out_file=dot_data, feature_names=feature_names,
                        class_names=class_names, filled=True)

        dot_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        node_attributes = {node.obj_dict['name']: {
            'label': node.get_label().replace('\\n', '<br>').replace('"', ''),
            'color': node.get_label().replace('"', '').split('class')[-1].split(' ')[-1]}
            for node in dot_graph.get_nodes() if node.get_label()}

        edge_list = [edge.obj_dict['points'] for edge in dot_graph.get_edge_list()]

        nx_graph = nx.from_edgelist(edgelist=edge_list)

        pos = nx.drawing.nx_pydot.graphviz_layout(nx_graph, prog='dot')

        nx.set_node_attributes(nx_graph, node_attributes)

        cls_handler = cls(graph=nx_graph, node_data_key='label', **params)

        cls_handler.build_graph_figure(pos=pos)

        return cls_handler

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
            if 'color' in node_data.keys():
                node_trace['marker']['color'] += tuple([hash(node_data['color']) % 256])
            else:
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

        else:
            self.graph_figure.layout = figure_layout

        iplot(self.graph_figure)

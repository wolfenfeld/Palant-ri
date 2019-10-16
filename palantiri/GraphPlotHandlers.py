from io import StringIO
import networkx as nx
import numpy as np
import pydotplus
from sklearn.tree import export_graphviz
import igraph as ig
import plotly.graph_objs as go
from plotly.offline import iplot

from palantiri.BasePlotHandlers import PlotHandler


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

    @staticmethod
    def _extract_graph_with_attributes_and_positions_from_tree(decision_tree, feature_names):
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

        return nx_graph, pos

    @classmethod
    def from_decision_tree(cls, decision_tree, feature_names, **params):
        """
        Handler from Decision tree
        :param decision_tree:  sklearn decision tree
        :param feature_names: the feature names
        :param params: other params
        :return: graph plot handler for the decision tree.
        """
        graph, pos = cls._extract_graph_with_attributes_and_positions_from_tree(decision_tree, feature_names)

        cls_handler = cls(graph=graph, node_data_key='label', **params)

        cls_handler.build_graph_figure(pos=pos)

        return cls_handler

    @staticmethod
    def _extract_graph_from_tree(decision_tree, feature_names):
        dot_data = StringIO()

        class_names = [str(name) for name in decision_tree.classes_]

        export_graphviz(decision_tree, out_file=dot_data, feature_names=feature_names,
                        class_names=class_names, filled=True)

        dot_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        edge_list = [edge.obj_dict['points'] for edge in dot_graph.get_edge_list()]

        nx_graph = nx.from_edgelist(edgelist=edge_list)

        return nx_graph

    @staticmethod
    def _generate_circular_layout(graph):
        g = ig.Graph(directed=True)
        g.add_vertices(graph.nodes())
        g.add_edges(graph.edges())
        return dict(zip(graph.nodes(), list(g.layout('rt_circular'))))

    @staticmethod
    def _build_edges_shapes(edges, layout):

        def build_edge_shape(edge, edges_layout):

            x0, y0 = edges_layout[edge[0]]
            r0 = np.sqrt(x0 ** 2 + y0 ** 2)
            x3, y3 = edges_layout[edge[1]]
            r3 = np.sqrt(x3 ** 2 + y3 ** 2)
            r = 0.5 * (r0 + r3)

            if r0 == 0 or r3 == 0:
                x1 = x0
                y1 = y0
                x2 = x3
                y2 = 0.2 * y0 + y3 * 0.8
            else:
                x1 = r * x0 / r0
                y1 = r * y0 / r0
                x2 = r * x3 / r3
                y2 = r * y3 / r3

            return dict(type='path',
                        layer='below',
                        # def SVG Bezier path representing an edge
                        path=f'M{x0} {y0}, C {x1} {y1}, {x2} {y2}, {x3} {y3}',
                        line=dict(color='rgb(210,210,210)', width=1))

        return [build_edge_shape(edge, layout) for edge in edges]

    @classmethod
    def from_forest(cls, list_of_trees, feature_names, **params):

        forest = nx.DiGraph()
        forest.add_nodes_from(nodes_for_adding=['Root'])

        for i, tree_ in enumerate(list_of_trees):

            graph = cls._extract_graph_from_tree(tree_, feature_names)

            rename_map = {node: node+'_{0}'.format(i) for node in graph.nodes()}

            forest = nx.union(forest, nx.relabel_nodes(graph, rename_map))

            forest.add_edge('Root', '0_{0}'.format(i))

        rename_map = {node: j for j, node in enumerate(forest.nodes())}

        forest = nx.relabel_nodes(forest, rename_map)

        nx.set_node_attributes(forest, {node: {'color': 'red'} for node in forest.nodes()})

        forest_layout = cls._generate_circular_layout(forest)

        edges_shapes = cls._build_edges_shapes(forest.edges, forest_layout)

        title = "Circular Tree"
        width = 800
        height = 775

        layout = go.Layout(title=title,
                           font=dict(size=12),
                           showlegend=False,
                           autosize=False,
                           width=width,
                           height=height,
                           xaxis=dict(visible=False),
                           yaxis=dict(visible=False),
                           hovermode='closest',
                           plot_bgcolor='rgb(10,10,10)',
                           shapes=edges_shapes)

        cls_handler = cls(graph=forest,  **params)

        cls_handler.build_graph_figure(pos=forest_layout, figure_layout=layout)

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
            node_trace['text'] += tuple([node_data[self.node_data_key] if self.node_data_key else node])
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
            figure_layout = go.Layout(title=dict(text='Graph Plot', x=0.5),
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


if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    import networkx as nx
    from sklearn import tree
    from sklearn.datasets import load_wine

    # load dataset
    data = load_wine()

    # feature matrix
    X = data.data

    # target vector
    y = data.target

    # class labels
    labels = data.feature_names
    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42
    )

    clf.fit(X, y)
    handler = GraphPlotHandler.from_forest(list_of_trees=clf.estimators_, feature_names=labels)
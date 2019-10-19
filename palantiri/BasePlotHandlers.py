from plotly.offline import plot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


class PlotHandler(object):
    """ Base class for the plot handlers """

    def __init__(self, **params):
        """
        The initialization function.
        :param params: parameters.
        """

        for k, v in params.items():
            setattr(self, k, v)

    @staticmethod
    def save_figure(figure, file_name):
        """
        Saving the figure as an html file.
        :param figure: the figure to be saved.
        :param file_name: The html file name
        """

        with open(file_name, "w") as text_file:
            text_file.write('< script src = "https://cdn.plot.ly/plotly-latest.min.js" > < / script > \n')
            text_file.write(plot(figure, include_plotlyjs=False, output_type='div'))

    @staticmethod
    def build_hexbin_figure(x, y):

        def get_hexbin_attributes(hexbin):
            paths = hexbin.get_paths()
            points_codes = list(paths[0].iter_segments())
            prototypical_hexagon = [item[0] for item in points_codes]
            return prototypical_hexagon, hexbin.get_offsets(), hexbin.get_facecolors(), hexbin.get_array()

        def make_hexagon(prototypical_hex, offset, fillcolor, linecolor=None):

            new_hex_vertices = [vertex + offset for vertex in prototypical_hex]
            vertices = np.asarray(new_hex_vertices[:-1])
            # hexagon center
            center = np.mean(vertices, axis=0)
            if linecolor is None:
                linecolor = fillcolor
            # define the SVG-type path:
            path = 'M '
            for vert in new_hex_vertices:
                path += f'{vert[0]}, {vert[1]} L'
            return dict(type='path',
                        line=dict(color=linecolor,
                                  width=0.5),
                        path=path[:-2],
                        fillcolor=fillcolor,
                        ), center

        def pl_cell_color(face_colors):

            return [f'rgb({int(R * 255)}, {int(G * 255)}, {int(B * 255)})' for (R, G, B, A) in face_colors]

        def build_plot_data(x_coordinates, y_coordinates):
            hexbin = plt.hexbin(x_coordinates, y_coordinates, gridsize=25, mincnt=0)

            hexagon_vertices, offsets, mpl_facecolors, bin_count = get_hexbin_attributes(hexbin)
            cell_color = pl_cell_color(mpl_facecolors)

            shape_list = []
            centers = []
            for k in range(len(offsets)):
                shape, center = make_hexagon(hexagon_vertices, offsets[k], cell_color[k])
                shape_list.append(shape)
                centers.append(center)

            centers_x, centers_y = zip(*centers)

            # define  text to be  displayed on hovering the mouse over the cells
            text = [f'x: {round(centers_x[k], 2)}<br>y: {round(centers_y[k], 2)}<br>counts: {int(counts[k])}'
                    for k in range(len(centers_y))]

            return centers_x, centers_y, shape_list, text, bin_count

        X, Y, shapes, hover_text, counts = build_plot_data(x, y)

        trace = go.Scatter(
            x=list(X),
            y=list(Y),
            mode='markers',
            marker=dict(size=0.5,
                        color=counts,
                        showscale=True,
                        colorbar=dict(
                            thickness=20,
                            ticklen=4
                        )),
            text=hover_text,
            hoverinfo='text'
        )

        axis = dict(showgrid=False,
                    showline=False,
                    zeroline=False,
                    ticklen=4
                    )

        layout = go.Layout(title='Hexbin plot',
                           width=530, height=550,
                           xaxis=axis,
                           yaxis=axis,
                           hovermode='closest',
                           shapes=shapes)

        return go.Figure(data=[trace], layout=layout)

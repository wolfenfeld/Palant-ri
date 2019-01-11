from plotly.offline import plot


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

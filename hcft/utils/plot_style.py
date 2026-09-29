import matplotlib.pyplot as plt
import numpy as np

plot_colors = ['#000000',
               '#bc2122',
               '#1f77b4',
               '#ff7f0e',
               '#2ca02c',
               '#9467bd',
               '#8c564b',
               '#e377c2',
               '#7f7f7f',
               '#17becf']
i_color = 0


def set_latex_mpl_format(font_size=15):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = font_size
    # To have math like LaTeX
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'

    # To save svg fonts as fonts and not paths
    plt.rcParams['svg.fonttype'] = 'none'


def get_color():
    global i_color
    color = plot_colors[i_color] if i_color < len(plot_colors) else np.random.rand(3)
    i_color += 1
    return color

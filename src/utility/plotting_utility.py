import matplotlib.pyplot as plt
import numpy as np

def set_matplotlib_global_properties():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 11
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

def get_default_figure_size():
    px = 1/plt.rcParams['figure.dpi'] # pixel in inches
    golden_ratio = (1+np.sqrt(5))/2
    figsize_x = 550 * px
    figsize_y = figsize_x/golden_ratio
    return (figsize_x, figsize_y)

def get_default_colors():
    colors_blue = ['#eff3ff', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']
    colors_orange = ['#feedde', '#fdbe85', '#fd8d3c', '#e6550d', '#a63603']
    colors_red = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
    colors_green = ['#edf8e9', '#bae4b3', '#74c476', '#31a354', '#006d2c']
    colors_gray = ['#f7f7f7', '#cccccc', '#969696', '#636363', '#252525']
    colors_purple = ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f']
    return {
        "disentangle_renyi_evenbly_vidal": colors_blue,
        "disentangle_renyi_cg": colors_orange,
        "disentangle_renyi_trm": colors_green,
        "disentangle_renyi_approx_cg": colors_orange,
        "disentangle_renyi_approx_trm": colors_green
    }

def save_data_for_plot(filename, columns, data):
    with open(filename, "w") as file:
        file.write(" ".join(columns) + "\n")
        for line in data:
            file.write(" ".join([str(x) for x in line]) + "\n")

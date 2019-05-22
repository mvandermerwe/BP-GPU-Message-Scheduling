#
# Created by Mark Van der Merwe
#

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

def load_data_from_files(files):
    '''
    Return array of data for each file.
    '''
    xs = []
    ys = []
    for fname in files:
        x, y = load_data_from_file(fname)
        if x is not None or y is not None:
            xs.append(x)
            ys.append(y)
    
    return xs, ys

def load_data_from_file(filename):
    '''
    Load data from the given file into a numpy array.
    Returns x and y arrays.
    '''
    try:
        return np.loadtxt(fname=filename, delimiter=",", unpack=True)
    except:
        return None, None

def plot_data_lines(axes, xs, ys, names):
    '''
    Plot the given data sets on the given axes.
    '''

    for x,y,label in zip(xs, ys, names):
        axes.plot(x, y, label=label)

#parser = ArgumentParser("Generate plot for the given list of files.")
#parser.add_argument("-f", "--files", nargs='+', help='List of files to plot together.', required=True)
#parser.add_argument("-n", "--names", nargs='+', help='List of labels to files.', required=True)
#parser.add_argument("-t", "--title", help='Title for plot', required=True)
#parser.add_argument("-x", "--xaxis", help='Label for x-axis.', required=True)
#parser.add_argument("-y", "--yaxis", help='Label for y-axis.', required=True)
#parser.add_argument("-xl", "--xstart", help='Start point for x-axis.', default=None, type=int)
#parser.add_argument("-xr", "--xend", help='End point for x-axis.', default=None, type=int)
#args = parser.parse_args()

#files = args.files
#names = args.names
#title = args.title
#xlabel = args.xaxis
#ylabel = args.yaxis
#xstart = args.xstart
#xend = args.xend

def plot_data(files, names, title, xlabel, ylabel, xstart, xend):
    xvals, yvals = load_data_from_files(files)
    fig, ax = plt.subplots(1, 1)
    plot_data_lines(ax, xvals, yvals, names)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xstart is not None and xend is not None:
        ax.set_xlim(left=int(xstart), right=int(xend))

    plt.title(title)
    fig.canvas.set_window_title(title)
    plt.show()

#plot_data(files, names, title, xlabel, ylabel, xstart, xend)

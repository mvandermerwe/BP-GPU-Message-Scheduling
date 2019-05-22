#
# Created by Mark Van der Merwe, Fall 2018
#

import os
import plot_data
from argparse import ArgumentParser
import pdb

parser = ArgumentParser("Plot all points for given test parameters together.")
parser.add_argument("-r", "--runtimes", nargs="+", help="Runtimes to plot (i.e., some or all of Loopy, RBP, RS, HBP).", required=True)
parser.add_argument("-tt", "--testtype", help="The type of test to plot data for (i.e., one of Ising, Chain, or Random).", required=True)
parser.add_argument("-i", "--info", help="Which type of plot to make (i.e., Iterations or Runtime).", required=True)
parser.add_argument("-t", "--title", help="Title for graph.", required=True)
parser.add_argument("-hbp", "--hbpruns", nargs="+", help="Runtime settings for HBP", default=[])
parser.add_argument("-xs", "--xstart", help="Start value for x axis.", default=None)
parser.add_argument("-xe", "--xend", help="End value for x axis.", default=None)
args = parser.parse_args()

runtimes = args.runtimes
testtype = args.testtype
info = args.info
title = args.title
hbpruns = args.hbpruns
xstart = args.xstart
xend = args.xend

files = []
names = []

for runtime in runtimes:
    folder = "Results" + str(runtime) + str(testtype) + "/"

    p_runs = []
    if runtime == "Hbp":
        for hbp_run in hbpruns:
            p_runs.extend(list(map(lambda x: hbp_run + "/" + x, next(os.walk(folder + hbp_run))[1])))
    else:
        p_runs = next(os.walk(folder))[1]

    for p_run in p_runs:
        run_file = folder + p_run + "/"

        if info == "Iterations":
            run_file += "iterations_cumulative.txt"
        else:
            run_file += "runtime_cumulative.txt"

        files.append(run_file)
        names.append(str(runtime) + " " + str(p_run))

plot_data.plot_data(files, names, title, info, "Cumulative Convergence", xstart, xend)

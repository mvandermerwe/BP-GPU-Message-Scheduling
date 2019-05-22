#
# Created by Mark Van der Merwe, Summer 2018
#

# Simple python script to generate ising model graphs in our encoding format.

from argparse import ArgumentParser
import random
import math

# Helper function that creates the string for an edge.
def edge_string(node_source, node_dest, ising_file):
    # 4 items for the potential function representing - (0,0),(0,1),(1,0),(1,1). However we really have two values - one for when the two values are the same, one for when they're different.
    # Start by sampling a random lambda in the [-0.5,0.5] range.
    lam = random.uniform(-0.5,0.5)
    same_potential = math.exp(lam * C)
    diff_potential = math.exp(-lam * C)
    # Now write the edge.
    ising_file.write(node_source + "," + node_dest + ",{" + str(same_potential) + "," + str(diff_potential) + "," + str(diff_potential) + "," + str(same_potential) + "}\n")

parser = ArgumentParser("Generate an ising model of specified sizes - sizes n1 to n2 by increment i.")
parser.add_argument("n1", help="Specify smallest grid size n1 for grid sizes.", type=int)
parser.add_argument("n2", help="Specify largest grid size n2 for grid sizes.", type=int)
parser.add_argument("i", help="Specify the increment for the grid sizes.", type=int)
parser.add_argument("-c", help="Specify the difficulty of the ising grids.", type=float, default=3.0)
parser.add_argument("-nb", help="Specify the bottom for range of grids to make.", type=int, default=1)
parser.add_argument("-ne", help="Specify the top for range of grids to make.", type=int, default=2)
args = parser.parse_args()

n1 = args.n1
n2 = args.n2
i = args.i
C = args.c
nb = args.nb
ne = args.ne

random.seed(42)

# Now, iterate through the sizes specified and create a random ising model grid (as specified in the RBP paper).
# Specifically:
#  Uniformly sample univariate potentials in the [0,1] range.
#  Pairwise potentials are e^lambda*C when x_i==x_j and e^-lambda*C, where lambda is sampled in the [-0.5,0.5] range.
for size in range(n1,n2+1,i):
    for grid_num in range(nb, ne):
        # Create the file for this size.
        filename = "ising_" + str(size) + "_" + str(grid_num) + ".txt"
        ising_file = open(filename, "w")
    
        # Start by initializing the nodes (all are from a two-element categorical distribution - either -1 or 1 corresponding to spin).
        ising_file.write("Nodes:\n")
        for x in range(0,size):
            for y in range(0,size):
                node_name = "node_" + str(x) + "_" + str(y)
                univariate_potential = random.uniform(0,1)
                ising_file.write(node_name + ",{" + str(univariate_potential) + "," + str(1.0-univariate_potential) + "}\n")
        ising_file.write("\n")
            
        # Next add the edges.
        ising_file.write("Edges:\n")
        for x in range(0,size-1):
            for y in range(0,size-1):
                # Each x,y node has one potential down and right to add.

                # Add right edge.
                node_source = "node_" + str(x) + "_" + str(y)
                node_dest = "node_" + str(x) + "_" + str(y+1)
                edge_string(node_source, node_dest, ising_file)
            
                # Add down edge.
                node_source = "node_" + str(x) + "_" + str(y)
                node_dest = "node_" + str(x+1) + "_" + str(y)
                edge_string(node_source, node_dest, ising_file)
        # The last row only has to draw the lines to the right.
        x = size-1
        for y in range(0,size-1):
            # Add right edge.
            node_source = "node_" + str(x) + "_" + str(y)
            node_dest = "node_" + str(x) + "_" + str(y+1)
            edge_string(node_source, node_dest, ising_file)

        # The last column only has to draw the lines down.
        y = size-1
        for x in range(0,size-1):
            # Add down edge.
            node_source = "node_" + str(x) + "_" + str(y)
            node_dest = "node_" + str(x+1) + "_" + str(y)
            edge_string(node_source, node_dest, ising_file)
        ising_file.write("\n")
            
        # We want the marginals for all nodes (for now, might change this later).
        ising_file.write("Marginals:\n")
        for x in range(0,size):
            for y in range(0,size):
                node_name = "node_" + str(x) + "_" + str(y)
                ising_file.write(node_name + ",")
        ising_file.write("\n")

        # We've finished this ising grid. Close the file.
        ising_file.close()
            

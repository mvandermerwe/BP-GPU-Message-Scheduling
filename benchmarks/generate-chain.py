#
# Created by Mark Van der Merwe, Summer 2018
#

# Simple python script to generate chain model graphs in our encoding format.

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

parser = ArgumentParser("Generate chain models of specified sizes - sizes n1 to n2 by increment i.")
parser.add_argument("n1", help="Specify smallest chain size n1.", type=int)
parser.add_argument("n2", help="Specify largest chain size n2.", type=int)
parser.add_argument("i", help="Specify the increment for the chain sizes.", type=int)
parser.add_argument("-c", help="Specify the difficulty of the chain grids.", type=int, default=3)
parser.add_argument("-n", help="Specify the number of chains of each size to make.", type=int, default=1)
args = parser.parse_args()

n1 = args.n1
n2 = args.n2
i = args.i
C = args.c
n = args.n

# Now, iterate through the sizes specified and create a random ising model grid (as specified in the RBP paper).
# Specifically:
#  Uniformly sample univariate potentials in the [0,1] range.
#  Pairwise potentials are e^lambda*C when x_i==x_j and e^-lambda*C, where lambda is sampled in the [-0.5,0.5] range.
for size in range(n1,n2+1,i):
    for chain_num in range(n):
        # Create the file for this size.
        filename = "chain_" + str(size) + "_" + str(chain_num) + ".txt"
        chain_file = open(filename, "w")
    
        # Start by initializing the nodes (all are from a two-element categorical distribution - either -1 or 1).
        chain_file.write("Nodes:\n")
        for x in range(0,size):
            node_name = "node_" + str(x)
            univariate_potential = random.uniform(0,1)
            chain_file.write(node_name + ",{" + str(univariate_potential) + "," + str(1.0-univariate_potential) + "}\n")
        chain_file.write("\n")
        
        # Next add the edges.
        chain_file.write("Edges:\n")
        for x in range(0,size-1):
            # Each node has one edge to the right to add.

            # Add right edge.
            node_source = "node_" + str(x)
            node_dest = "node_" + str(x+1)
            edge_string(node_source, node_dest, chain_file)
        chain_file.write("\n")
            
        # We want the marginals for all nodes.
        chain_file.write("Marginals:\n")
        for x in range(0,size):
            node_name = "node_" + str(x)
            chain_file.write(node_name + ",")
        chain_file.write("\n")

        # We've finished this chain graph. Close the file.
        chain_file.close()
            

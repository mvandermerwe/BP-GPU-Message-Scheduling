# Message Scheduling for Performant, Many-Core Belief Propagation

This repository holds source code for the paper "Message Scheduling for Performant, Many-Core Belief Propagation." We provide our implementations of Randomized Belief Propagation (RnBP), our main contribution from this work, as well as Loopy Belief Propagation (LBP), Residual Belief Propagation (RBP), and Residual Splash (RS) on the GPU, and serial LBP, RBP, and Variable Elimination for comparison.

## Dependencies

Our work depends on the following:

* gcc/5.4.0
* cuda/9.1
* boost/1.66.0
* [cuRAND](https://developer.nvidia.com/curand)
* [CUB](https://nvlabs.github.io/cub/)

Other versions of GCC, CUDA, and Boost may work, but not confirmed.

Set your CUDA_INC, CUDA_LIB, BOOST_INC, BOOST_LIB environment variables to the include and library directories for CUDA/Boost. For CUB, clone the project from their [site](https://nvlabs.github.io/cub/) and link the `cub/` folder in that repository via the environment variable CUB_HOME.

## Building and Running

We have two Makefiles, depending on whether you want to compile the GPU or CPU version of the algorithm.
The following shows how to get our GPU RnBP code running on some provided small graphs:

```
cd src
mv Makefile.parallel Makefile # Swap in Makefile.serial to build the CPU codes.

make SRC=main.cpp INFER=rnbp

# General structure is: ./main.out <test_file> <timeout (sec)> <alg. hyperparams (decribed below)>.
./main.out ../benchmarks/SmallGraphs/ising_3_0.txt 10 6 3
```

SRC can be set to main.cpp for running single examples or time_tests.cpp for doing more extensive timing tests (as run in the paper).

INFER can be set to:
* GPU (Makefile.parallel): loopy, rbp, rs, rnbp
* CPU (Makefile.serial): loopy, rbp, ve

loopy refers to LBP, rbp to RBP, rs to RS, rnbp to RnBP, ve to Variable Elimination (note: our variable elimination only works for binary ising grids and should not be used as a general VE algorithm).

For GPU RBP, RS, and RnBP, the algorithm relies on hyperparameters that determine runtime parallelism:
* For RBP/RS, you must provide the parallelism `p` that indicates the percentage of messages updated. This is passed as a value `a`, where `p=1/2^a`. Thus to run with 1/8 parallelism, one should run: `./main.out <test_file> <timeout> 3`.
* For RnBP, you must provide both the high and low parallelisms (see paper for details). This is passed as two values `a,b` where `LowP=a/10` and `HighP=b/10`. Thus to run with 0.3 low parallelism and 0.7 high parallelism, one should run: `./main.out <test_file> <timeout> 3 7`.

### Contact 
Mark Van der Merwe, mark.vandermerwe@utah.edu

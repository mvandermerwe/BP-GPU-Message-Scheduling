//
// Created by Mark Van der Merwe, Summer 2018
//

#ifndef CUPGM_PARALLEL_INFERENCE_INFERENCE_HELPERS_H_
#define CUPGM_PARALLEL_INFERENCE_INFERENCE_HELPERS_H_

#include "infer_data.h"
#include <curand.h>
#include <ctime>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

#define gpuCurandErrchk(ans) { gpuCurandAssert((ans), __FILE__, __LINE__); }
inline void gpuCurandAssert(curandStatus_t code, const char *file, int line, bool abort=true)
{
  if (code != CURAND_STATUS_SUCCESS) 
    {
      fprintf(stderr,"GPUassert: Curand error! %s %d\n", file, line);
      if (abort) exit(code);
    }
}

__host__ __device__ int ceiling_division(int x, int y);

// Use Euclidian distance as the metric to determine if the operations have converged.
__host__ __device__ double message_delta(double* edges, double* workspace, int edge_start);

__host__ __device__ void compute_message(device_graph d_graph, device_pgm d_pgm, int edge_id);

__device__ void compute_marginal(device_graph d_graph, device_pgm d_pgm, int* marginal_to_node_id, int* marginal_to_marginal_rep, double* marginals, int marginal);

__global__ void compute_marginals(device_graph d_graph, device_pgm d_pgm, int* marginal_to_node_id, int* marginal_to_marginal_rep, double* marginals, int node_count);

int edges_to_update(int num_edges, double* d_residuals, double cutoff);
int edges_to_update(int num_edges, double* d_residuals, double epsilon, double* d_rand_vals, double p);

std::vector<double> compute_marginals(pgm* pgm, device_graph d_graph, device_pgm d_pgm, bool verbose);

__host__ __device__ void print_doubles(double* values, int count);

__host__ __device__ void print_ints(int* values, int count);

void print_gpu_data(int* values, int count);

void print_gpu_data(double* values, int count);

void scale_launch_params(int* gridSize, int* blockSize, int problemSize);

#endif // CUPGM_PARALLEL_INFERENCE_INFERENCE_HELPERS_H_

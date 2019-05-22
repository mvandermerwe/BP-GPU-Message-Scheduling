//
// Created by Mark Van der Merwe, Summer 2018
//

#include "infer.h"
#include "infer_data.h"
#include "inference_helpers.h"
#include "../header.h"
#include "math_functions.h"
#include <stdio.h>
#include <cub.cuh>
#include <iostream>
#include "cuda_profiler_api.h"
#include <ctime>

__global__ void loopy_bp(device_graph d_graph, device_pgm d_pgm, int* need_update, int edge_count, double epsilon, int edge_size /* debug */) {
  
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  int num_not_converged = 0;

  // Iterate through all the edges assigned to this thread. This is likely as well balanced in terms of work as we can get.
  for (int edge_id = idx; edge_id < edge_count; edge_id += step) {
    compute_message(d_graph, d_pgm, edge_id);

    // Calculate local convergence - that is, in Euclidian space, see if the distance between the two messages is less than the provided threshold epsilon.
    int edges_index = d_pgm.edge_idx_to_edges_idx[edge_id];
    double difference = message_delta(d_pgm.edges, d_pgm.workspace, edges_index);

    // Return whether this message has converged.
    if (difference > epsilon) {
      num_not_converged += 1;
    }
  }

  // Track the number of non-converged edges for each thread.
  need_update[(blockIdx.x * blockDim.x) + threadIdx.x] = num_not_converged;
}

__global__ void setup_residuals(device_graph d_graph, device_pgm d_pgm, double* residuals, int num_edges) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  // Iterate through all the edges assigned to this thread. This is likely as well balanced in terms of work as we can get.
  for (int edge_id = idx; edge_id < num_edges; edge_id += step) {

    // One more computation.
    compute_message(d_graph, d_pgm, edge_id);

    // Calculate residual.
    double residual = message_delta(d_pgm.edges, d_pgm.workspace, d_pgm.edge_idx_to_edges_idx[edge_id]);

    // Place new residual.
    residuals[edge_id] = residual;
  }
}

std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, double epsilon, int timeout, std::vector<int> runtime_params, bool verbose) {

  //
  // Setup GPU Data.
  //

  // Move pgm data to the gpu.
  std::pair<device_graph, device_pgm> infer_data = setup_gpu_data(pgm);
  int num_edges = pgm->num_edges();
  if (verbose)
    std::cout << "Number of edges: " << num_edges << std::endl;

  // Create a convergence array that each thread can write to in order to determine overall convergence.
  int* d_need_update_sum;
  gpuErrchk(cudaMalloc((void**) &d_need_update_sum, 1 * sizeof(int)));

  int* h_need_update_sum = (int*) malloc(sizeof(int));
  *h_need_update_sum = num_edges;

  //
  // Setup GPU Runtime.
  //

  // Use cuda runtime to determine best kernel launch parameters.
  int blockSize;
  int minGridSize;
  int gridSize;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, loopy_bp, 0, 0));
  // Now we round up the output according to our array input size.
  gridSize = (num_edges + blockSize - 1) / blockSize;

  // We do this after runtime config due to sizing dependencies.
  int* d_need_update;
  gpuErrchk(cudaMalloc((void**) &d_need_update, gridSize * blockSize * sizeof(int)));

  //
  // Setup Runtime tracking.
  //

  bool converged = false;
  
  // Track convergence of edges by iteration and time.
  std::vector<std::pair<int, int>> edges_converged_iterations;
  std::vector<std::pair<float, int>> edges_converged_time;

  // Time our code.
  std::clock_t begin = std::clock();
  std::clock_t since;
  float time = 0.0;

  // Set up profiling events.
  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  gpuErrchk(cudaEventRecord(start));

  //
  // Run.
  //

  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_need_update, d_need_update_sum, num_edges);

  // Allocate temporary storage
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  int iterations = 0;
  while(!converged && time < timeout) {
    ++iterations;

    // Now we can launch the kernel:
    loopy_bp<<<gridSize, blockSize>>>(infer_data.first, infer_data.second, d_need_update, num_edges, epsilon, pgm->edges.size());

    // First we need to swap our edge pointer to point to our workspace, where the updates are currently written.
    double* new_edges = infer_data.second.workspace;
    infer_data.second.workspace = infer_data.second.edges;
    infer_data.second.edges = new_edges;

    // Use CUB to determine if we have converged.
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_need_update, d_need_update_sum, num_edges);

    // Copy results back to host so we know whether to run another iteration.
    gpuErrchk(cudaMemcpy(h_need_update_sum, d_need_update_sum, 1*sizeof(int), cudaMemcpyDeviceToHost));
    converged = *h_need_update_sum == 0;
    if (iterations % 5 == 1 && iterations < 1002) {
      edges_converged_iterations.push_back(std::make_pair(iterations, *h_need_update_sum));
      edges_converged_time.push_back(std::make_pair(time, *h_need_update_sum));
    }

    since = std::clock();
    time = float(since - begin) / CLOCKS_PER_SEC;
  }

  gpuErrchk(cudaEventRecord(stop));
  gpuErrchk(cudaEventSynchronize(stop));
  float milliseconds = 0;
  gpuErrchk(cudaEventElapsedTime(&milliseconds, start,stop));

  // Now the convergence should be complete, and we can launch a new kernel to determine the marginal distributions.
  std::vector<double> result = compute_marginals(pgm, infer_data.first, infer_data.second, verbose);
  if (verbose) {
    print_doubles(result.data(), result.size());
    std::cout << "Stopped after " << iterations << " iterations." << std::endl;
  }

  // Free up all our memory.
  free(h_need_update_sum);
  free_gpu_data(infer_data);
  gpuErrchk(cudaFree(d_need_update));
  gpuErrchk(cudaFree(d_need_update_sum));
  gpuErrchk(cudaFree(d_temp_storage));

  std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> results(converged ? milliseconds : -1.0, result, converged ? iterations : -1, edges_converged_iterations, edges_converged_time);
  return results;
}

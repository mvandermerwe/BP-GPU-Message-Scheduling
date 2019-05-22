//
// Created by Mark Van der Merwe, Fall 2018
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
#include <curand.h>
#include <ctime>
#include <cmath>
#include <utility>
#include <tuple>

// Random BP: update edges probabilistically. Start by updating all of them, then randomly ignore certain edges.

__global__ void random_update(device_graph d_graph, device_pgm d_pgm, double* residuals, double epsilon, float* rand_vals, float p, int num_edges) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  // Iterate through all the edges assigned to this thread. This is likely as well balanced in terms of work as we can get.
  for (int edge_id = idx; edge_id < num_edges; edge_id += step) {

    // Determine if this edge needs to be updated, that is, if the residual for this edge is greater than epsilon.
    if (residuals[edge_id] > epsilon) {
      if (rand_vals[edge_id] >= p) {
	compute_message(d_graph, d_pgm, edge_id);
      }
    }
  }
}

__global__ void update_residuals(device_graph d_graph, device_pgm d_pgm, double* residuals, int* need_update, double epsilon, int num_edges) {
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

    need_update[edge_id] = residual > epsilon ? 1 : 0;
  }
}

std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, double epsilon, int timeout, std::vector<int> runtime_params, bool verbose) {
  
  //
  // Setup GPU data.
  //

  std::pair<device_graph, device_pgm> infer_data = setup_gpu_data(pgm);

  int num_edges = pgm->num_edges();
  int edge_rep_size = pgm->edges.size();

  // Create a residual array. Each edge gets a residual, which is used to decide whether the edge needs to update.
  double* d_residuals;
  gpuErrchk(cudaMalloc((void**) &d_residuals, num_edges * sizeof(double)));
  std::vector<double> residuals_(num_edges, 10.0);
  gpuErrchk(cudaMemcpy(d_residuals, residuals_.data(), num_edges * sizeof(double), cudaMemcpyHostToDevice));
  
  // Track which edges need updates.
  int* d_need_update;
  gpuErrchk(cudaMalloc((void**) &d_need_update, num_edges * sizeof(int)));

  int* d_need_update_sum;
  gpuErrchk(cudaMalloc((void**) &d_need_update_sum, 1*sizeof(int)));
  int* need_update_sum = (int*) malloc(sizeof(int));
  
  // Random number generation.
  curandGenerator_t gen;
  gpuCurandErrchk(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

  float* d_rand_vals;
  gpuErrchk(cudaMalloc((void**) &d_rand_vals, num_edges * sizeof(float)));
  std::vector<float> rand_val_init(num_edges, 1.0);
  gpuErrchk(cudaMemcpy(d_rand_vals, rand_val_init.data(), num_edges * sizeof(float), cudaMemcpyHostToDevice));

  //
  // Setup GPU runtime.
  //

  int iterations = 0;
  bool converged = false;

  // The following decides parallelism for updates.
  float to_update_change = 1.0;
  float p;

  if (runtime_params.size() < 2) {
    std::cout << "RnBP requres: <low parallelism> <high parallelism>" << std::endl;
    return std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>>(0.0,{},0,{},{});
  }
  float low_parallelism = (float) runtime_params[0] / 10.0;
  float high_parallelism = (float) runtime_params[1] / 10.0;

  // Use cuda runtime to determine best kernel launch parameters.
  int blockSizeUpdate;
  int minGridSizeUpdate;
  int gridSizeUpdate;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSizeUpdate, &blockSizeUpdate, random_update, 0, 0));
  gridSizeUpdate = (num_edges + blockSizeUpdate - 1) / blockSizeUpdate;

  int blockSizeResidual;
  int minGridSizeResidual;
  int gridSizeResidual;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSizeResidual, &blockSizeResidual, update_residuals, 0, 0));
  gridSizeResidual = (num_edges + blockSizeResidual - 1) / blockSizeResidual;

  // Determine temporary device storage requirements for CUB convergence reductions.
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_need_update, d_need_update_sum, num_edges);

  // Allocate temporary storage
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  //
  // Setup Runtime tracking.
  //

  // Time our code.
  std::clock_t begin = std::clock();
  std::clock_t since;
  float time = 0.0;

  // Track convergence of edges by iteration and time.
  std::vector<std::pair<int, int>> edges_converged_iterations;
  std::vector<std::pair<float, int>> edges_converged_time;

  // Set up profiling events.
  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk(cudaEventRecord(start));
  
  //
  // Run.
  //

  while(!converged && time < timeout) {
    ++iterations;

    // Change p depending on change in number of updates to be done.
    // Note these are only chosen from amongst those with residual above epsilon.
    // to_update_change = current / old.
    p = to_update_change < 0.9 ? high_parallelism : low_parallelism;

    // int to_update_count = edges_to_update(num_edges, d_residuals, epsilon, d_rand_vals, p);
    // std::cout << iterations << ", " << to_update_count << std::endl;

    // Update edges as needed, i.e., if the edge residual is high enough.
    random_update<<<gridSizeUpdate, blockSizeUpdate>>>(infer_data.first, infer_data.second, d_residuals, epsilon, d_rand_vals, p, num_edges);

    // Copy workspace into edges - this insures our residual update is accurate.
    cudaMemcpy(infer_data.second.edges, infer_data.second.workspace, edge_rep_size * sizeof(double), cudaMemcpyDeviceToDevice);

    // Update residuals.
    update_residuals<<<gridSizeResidual, blockSizeResidual>>>(infer_data.first, infer_data.second, d_residuals, d_need_update, epsilon, num_edges);

    // Now copy edges into workspace - this insures our next round of updates is correct.
    cudaMemcpy(infer_data.second.workspace, infer_data.second.edges, edge_rep_size * sizeof(double), cudaMemcpyDeviceToDevice);

    // Use CUB to determine if we have converged.
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_need_update, d_need_update_sum, num_edges);

    // Copy results back to host so we know whether to run another iteration.
    int old_update_sum = *need_update_sum;
    gpuErrchk(cudaMemcpy(need_update_sum, d_need_update_sum, 1*sizeof(float), cudaMemcpyDeviceToHost));

    if (iterations % 5 == 1 && iterations < 1002) {
      edges_converged_iterations.push_back(std::make_pair(iterations,*need_update_sum));
      edges_converged_time.push_back(std::pair<float, int>(time, *need_update_sum));
    }

    to_update_change = (float) *need_update_sum / (float) old_update_sum;
    //std::cout << iterations << ", " << *need_update_sum << std::endl;
    if (*need_update_sum == 0) {
      converged = true;
    }

    // Generate random numbers to decide updates.
    gpuCurandErrchk(curandGenerateUniform(gen, d_rand_vals, num_edges));

    since = std::clock();
    time = float(since - begin) / CLOCKS_PER_SEC;
  }

  // if (!converged) {
  //   converged = serial_residual_belief_propagation(pgm, infer_data, num_edges, d_residuals, epsilon, begin, timeout);
  // }

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
  free_gpu_data(infer_data);
  gpuErrchk(cudaFree(d_residuals));
  gpuErrchk(cudaFree(d_need_update));
  gpuErrchk(cudaFree(d_need_update_sum));
  free(need_update_sum);
  gpuErrchk(cudaEventDestroy(start));
  gpuErrchk(cudaEventDestroy(stop));
  gpuErrchk(cudaFree(d_temp_storage));

  return std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>>(converged ? milliseconds : -1.0, result, converged ? iterations : -1, edges_converged_iterations, edges_converged_time);
}

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

__global__ void rbp_update(device_graph d_graph, device_pgm d_pgm, float* residuals, int* edge_ids, int* edges_effected, int edge_count, float epsilon, int edge_size /* debug */) {
  //----------------------//
  // Update first p edges //
  //----------------------//

  // Determine the edge this thread will be responsible for.
  int edge = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (edge < edge_count) {
    // Map the edge number to the actual edge id.
    int edge_id;
    edge_id = edge_ids[edge]; // This leads to very strided access (probably). Could build frontier here, then use another kernel to actually update - allowing for better coalesced accesses. See if needed.

    // Update that edge.
    compute_message(d_graph, d_pgm, edge_id);

    // Find all edges outgoing from my current edge and signify that they should be updated.
    int start_outgoing_idx = d_graph.edge_idx_to_outgoing_edges[edge_id];
    int end_outgoing_idx = d_graph.edge_idx_to_outgoing_edges[edge_id + 1];
    for (int outgoing_idx = start_outgoing_idx; outgoing_idx < end_outgoing_idx; ++outgoing_idx) {
      edges_effected[d_graph.edge_outgoing_edges[outgoing_idx]] = 1;
    }

    // Of course, the current edge's residual should also be updated.
    edges_effected[edge_id] = 1;
  }
}

//----------------------//
//   Update residuals   //
//----------------------//
__global__ void rbp_residuals(device_graph d_graph, device_pgm d_pgm, float* residuals, int* edge_ids, int* edges_effected, int edge_count, float epsilon, int edge_size /* debug */) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  for (int edge_id = idx; edge_id < edge_count; edge_id += step) {
    // Determine if this thread needs updating:
    if (edges_effected[edge_id] == 1) {
      // Compute this edge one more time.
      compute_message(d_graph, d_pgm, edge_id);
      
      // Now calculate the residual (i.e., change in message).
      float residual = message_delta(d_pgm.edges, d_pgm.workspace, d_pgm.edge_idx_to_edges_idx[edge_id]);
      
      // Place the new residual into the residual array.
      residuals[edge_id] = residual;
    } 

    // Reset edges effected.
    edges_effected[edge_id] = 0;
  }
}

std::tuple<float, std::vector<float>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, float epsilon, int timeout, std::vector<int> runtime_params, bool verbose) {

  //
  // Setup GPU data.
  //

  std::pair<device_graph, device_pgm> infer_data = setup_gpu_data(pgm);

  int num_edges = pgm->num_edges();
  int edge_rep_size = pgm->edges.size();
  if (verbose)
    std::cout << "Number of edges: " << num_edges << std::endl;

  // Create a residual array. Each edge gets a residual, note we also have an edge id array that lets us do a key value sort to determine whose updating who.
  // At each round, to determine who to update, we perform a key-value sort and choose the top p keys.
  float* d_residuals;
  gpuErrchk(cudaMalloc((void**) &d_residuals, num_edges * sizeof(float)));
  std::vector<float> residuals_(num_edges, 10.0);
  gpuErrchk(cudaMemcpy(d_residuals, residuals_.data(), num_edges * sizeof(float), cudaMemcpyHostToDevice));
  
  float* d_residuals_out;
  gpuErrchk(cudaMalloc((void**) &d_residuals_out, num_edges * sizeof(float)));
  float* top_residual = (float*) malloc(sizeof(float));

  std::vector<int> edge_ids_;
  for (int i = 0; i < num_edges; ++i) {
    edge_ids_.push_back(i);
  }

  int* d_edge_ids;
  gpuErrchk(cudaMalloc((void**) &d_edge_ids, num_edges * sizeof(int)));
  gpuErrchk(cudaMemcpy(d_edge_ids, edge_ids_.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));

  // Indicate whether a given edge should be updated after other edges are updated.
  int* d_edge_effected;
  gpuErrchk(cudaMalloc((void**) &d_edge_effected, num_edges * sizeof(int)));
  std::vector<int> edge_effected_(num_edges, 0);
  gpuErrchk(cudaMemcpy(d_edge_effected, edge_effected_.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));

  //
  // Setup GPU runtime.
  //

  if (runtime_params.size() < 1) {
    std::cout << "RBP requires parallelism divisor a, where p = 1/2^a." << std::endl;
    return std::tuple<float, std::vector<float>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>>(0.0,{},0,{},{});
  }
  float p = 1.0 / (float) std::pow(2, runtime_params[0]);

  // Use cuda runtime to determine best kernel launch parameters.
  int blockSizeUpdate;
  int minGridSizeUpdate;
  int gridSizeUpdate;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSizeUpdate, &blockSizeUpdate, rbp_update, 0, 0));
  // Now we round up the output according to our array input size. Note, we scale this by the provided p.
  // This allows us to update a smaller selection of the edges in order to exploit the advantages provided.
  int edges_to_update = (int) (num_edges * p);
  edges_to_update = edges_to_update == 0 ? 1 : edges_to_update;

  if (edges_to_update < blockSizeUpdate) {
    blockSizeUpdate = edges_to_update;
    gridSizeUpdate = 1;
  } else {
    gridSizeUpdate = (edges_to_update + blockSizeUpdate - 1) / blockSizeUpdate;
  }

  std::vector<int> edge_ids_out_;
  for (int i = 0; i < num_edges; i += ceiling_division(num_edges, edges_to_update)) {
    edge_ids_out_.push_back(i);
  }
  int i = 1;
  while(edge_ids_out_.size() < num_edges) {
    edge_ids_out_.push_back(i);
    ++i;
    if (i % ceiling_division(num_edges, edges_to_update) == 0)
      ++i;
  }

  // This GPU data is setup here since it depends on the runtime settings.
  int* d_edge_ids_out;
  gpuErrchk(cudaMalloc((void**) &d_edge_ids_out, num_edges * sizeof(int)));
  gpuErrchk(cudaMemcpy(d_edge_ids_out, edge_ids_out_.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));

  int blockSizeResidual;
  int minGridSizeResidual;
  int gridSizeResidual;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSizeResidual, &blockSizeResidual, rbp_residuals, 0, 0));
  gridSizeResidual = (num_edges + blockSizeResidual - 1) / blockSizeResidual;

  // Determine temporary device storage requirements for sorting.
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_residuals, d_residuals_out, d_edge_ids, d_edge_ids_out, num_edges);
  // Allocate temporary storage
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  //
  // Setup runtime tracking.
  //

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

  int iterations = 0;
  bool converged = false;

  while(!converged && time < timeout) {
    ++iterations;

    // Update messages.
    rbp_update<<<gridSizeUpdate, blockSizeUpdate>>>(infer_data.first, infer_data.second, d_residuals, d_edge_ids_out, d_edge_effected, num_edges, epsilon, pgm->edges.size());

    // Copy workspace into edges - this insures our residual update is accurate.
    cudaMemcpy(infer_data.second.edges, infer_data.second.workspace, edge_rep_size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Update residuals.
    rbp_residuals<<<gridSizeResidual, blockSizeResidual>>>(infer_data.first, infer_data.second, d_residuals, d_edge_ids_out, d_edge_effected, num_edges, epsilon, pgm->edges.size());

    // Now copy edges into workspace - this insures our next round of updates is correct.
    cudaMemcpy(infer_data.second.workspace, infer_data.second.edges, edge_rep_size * sizeof(float), cudaMemcpyDeviceToDevice);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_residuals, d_residuals_out, d_edge_ids, d_edge_ids_out, num_edges);

    // Check largest residual. If it's less than epsilon, we know we've converged.
    gpuErrchk(cudaMemcpy(top_residual, d_residuals_out, sizeof(float), cudaMemcpyDeviceToHost)); // Only copy back the first value.
    converged = *top_residual < epsilon;

    since = std::clock();
    time = float(since - begin) / CLOCKS_PER_SEC;
  }

  gpuErrchk(cudaEventRecord(stop));
  gpuErrchk(cudaEventSynchronize(stop));
  float milliseconds = 0;
  gpuErrchk(cudaEventElapsedTime(&milliseconds, start,stop));

  // Now the convergence should be complete, and we can launch a new kernel to determine the marginal distributions.
  std::vector<float> result = compute_marginals(pgm, infer_data.first, infer_data.second, verbose);
  if (verbose) {
    print_floats(result.data(), result.size());
    std::cout << "Stopped after " << iterations << " iterations." << std::endl;
  }

  // Free up all our memory.
  free(top_residual);
  free_gpu_data(infer_data);
  gpuErrchk(cudaFree(d_temp_storage));
  gpuErrchk(cudaFree(d_residuals));
  gpuErrchk(cudaFree(d_residuals_out));
  gpuErrchk(cudaFree(d_edge_ids));
  gpuErrchk(cudaFree(d_edge_effected));
  gpuErrchk(cudaFree(d_edge_ids_out));
  gpuErrchk(cudaEventDestroy(start));
  gpuErrchk(cudaEventDestroy(stop));

  return std::tuple<float, std::vector<float>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>>(converged ? milliseconds : -1.0, result, converged ? iterations : -1, {}, {});
}

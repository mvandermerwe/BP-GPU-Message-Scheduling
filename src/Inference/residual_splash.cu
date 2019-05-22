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
#include <cmath>
#include "cuda_profiler_api.h"
#include <ctime>

// edge_frontier - the next frontier of edges to update for the ith step.
// edges_effected - the total frontier of updated edges for this round.
__device__ void advance_edge_frontier(int* edge_frontier, int* edges_effected, int* nodes_effected, int* item_to_outgoing_idx, int* outgoing, int id, int* edge_idx_to_dest_node_idx, bool include_inverse) {
  int start_outgoing_idx = item_to_outgoing_idx[id];
  int end_outgoing_idx = item_to_outgoing_idx[id + 1];
  for (int outgoing_idx = start_outgoing_idx; outgoing_idx < end_outgoing_idx; ++outgoing_idx) {
    int edge_id = outgoing[outgoing_idx];
    edge_frontier[edge_id] = 1;

    // For edges effected, we want to also add in our inverse edge.
    if (include_inverse) {
      int inverse_edge_id = edge_id % 2 == 0 ? edge_id + 1 : edge_id - 1;
      edges_effected[inverse_edge_id] = 1;
    }
    edges_effected[edge_id] = 1;
    nodes_effected[edge_idx_to_dest_node_idx[edge_id]] = 1;
  }
}

__global__ void rs_node_select(device_graph d_graph, device_pgm d_pgm, int* node_ids, int* nodes_effected, int* frontier, int* edges_effected, int node_count, int edge_count) {

  // Determine the node this thread will be responsible for.
  int node = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (node < node_count) {
    // Map the node number to the actual node id.
    int node_id = node_ids[node];

    // Determine the edges outgoing from this node.
    advance_edge_frontier(frontier, edges_effected, nodes_effected, d_graph.node_idx_to_outgoing_edges, d_graph.node_outgoing_edges, node_id, d_graph.edge_idx_to_dest_node_idx, true);

    // Mark this node as needing it's residual updated.
    nodes_effected[node_id] = 1;
  }
}

__global__ void rs_generate_next_frontier(device_graph d_graph, device_pgm d_pgm, int* nodes_effected, int* last_frontier, int* frontier, int* edges_effected, int edge_count, bool include_inverse) {
  
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  for (int edge_id = idx; edge_id < edge_count; edge_id += step) {
    if (last_frontier[edge_id] == 1) {
      // Extend the edges effected to include this one's neighbors.
      advance_edge_frontier(frontier, edges_effected, nodes_effected, d_graph.edge_idx_to_outgoing_edges, d_graph.edge_outgoing_edges, edge_id, d_graph.edge_idx_to_dest_node_idx, include_inverse);
    }
  }
}

__global__ void rs_calculate_updates(device_graph d_graph, device_pgm d_pgm, int* frontier, int edge_count, bool forward) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  for (int edge_id = idx; edge_id < edge_count; edge_id += step) {
    if (frontier[edge_id] == 1) {
      int actual_edge_id;
      if (!forward) {
	// If we are doing the message collection phase, we need to swap the edge id to be the opposite direction edge.
	actual_edge_id = edge_id % 2 == 0 ? edge_id + 1 : edge_id - 1;
      } else {
	actual_edge_id = edge_id;
      }

      // Compute the edge.
      compute_message(d_graph, d_pgm, actual_edge_id);
    }
  }
}

__global__ void rs_compute_edge_residuals(device_graph d_graph, device_pgm d_pgm, float* edge_residuals, int* edges_effected, int edge_count) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  for (int edge_id = idx; edge_id < edge_count; edge_id += step) {
    if (edges_effected[edge_id] == 1) {
      // Compute the edge once more. TODO: Cache updates.
      compute_message(d_graph, d_pgm, edge_id);
      
      // Now compute the residual from this.
      float edge_residual = message_delta(d_pgm.edges, d_pgm.workspace, d_pgm.edge_idx_to_edges_idx[edge_id]);
      edge_residuals[edge_id] = edge_residual;
    }

    // Clear edges effected.
    edges_effected[edge_id] = 0;
  }
}

__global__ void rs_compute_node_residuals(device_graph d_graph, device_pgm d_pgm, float* node_residuals, float* edge_residuals, int* nodes_effected, int node_count) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  for (int node_id = idx; node_id < node_count; node_id += step) {
    if (nodes_effected[node_id] == 1) {
      // Compute residual of this node.
      // Residual is defined as the max of the residuals of the incoming messages.
      float residual = 0.0;
      
      int start_incoming_idx = d_graph.node_idx_to_incoming_edges[node_id];
      int end_incoming_idx = d_graph.node_idx_to_incoming_edges[node_id + 1];
      for (int incoming_idx = start_incoming_idx; incoming_idx < end_incoming_idx; ++incoming_idx) {
	float edge_residual = edge_residuals[d_graph.node_incoming_edges[incoming_idx]];
	if (edge_residual > residual)
	  residual = edge_residual;
      }

      node_residuals[node_id] = residual;
    }

    // Clear effected nodes.
    nodes_effected[node_id] = 0;
  }
}

std::tuple<float, std::vector<float>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, float epsilon, int timeout, std::vector<int> runtime_params, bool verbose) {

  //
  // Setup GPU data.
  //

  std::pair<device_graph, device_pgm> infer_data = setup_gpu_data(pgm);

  int num_edges = pgm->num_edges();
  int num_nodes = pgm->pgm_graph->node_idx_to_incoming_edges.size() - 1;
  int edge_rep_size = pgm->edges.size();
  if (verbose) {
    std::cout << "Number of edges: " << num_edges << std::endl;
    std::cout << "Number of nodes: " << num_nodes << std::endl;
  }

  // Size of each splash.
  int h = 2;

  // Create a residual array - each node gets a residual.
  // At each round, to determine who to update, we perform a key-value sort and choose the top p keys.
  float* d_node_residuals;
  gpuErrchk(cudaMalloc((void**) &d_node_residuals, num_nodes * sizeof(float)));
  std::vector<float> node_residuals_(num_nodes, 10.0); // Start all node_residuals as 10.0, so all nodes eventually update.
  gpuErrchk(cudaMemcpy(d_node_residuals, node_residuals_.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice));

  float* d_node_residuals_out;
  gpuErrchk(cudaMalloc((void**) &d_node_residuals_out, num_nodes * sizeof(float)));
  float* top_residual = (float*) malloc(sizeof(float));
  std::vector<int> node_ids_;
  for (int i = 0; i < num_nodes; ++i) {
    node_ids_.push_back(i);
  }
  int* d_node_ids;
  gpuErrchk(cudaMalloc((void**) &d_node_ids, num_nodes * sizeof(int)));
  gpuErrchk(cudaMemcpy(d_node_ids, node_ids_.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));

  // We also need residuals for our edges, which are used to compute the node residuals.
  float* d_edge_residuals;
  gpuErrchk(cudaMalloc((void**) &d_edge_residuals, num_edges * sizeof(float)));
  std::vector<float> edge_residuals_(num_edges, 10.0); // Start all edge_residuals as 10.0, so all edges eventually update.
  gpuErrchk(cudaMemcpy(d_edge_residuals, edge_residuals_.data(), num_edges * sizeof(float), cudaMemcpyHostToDevice));

  // Indicate whether a given node's residual should be updated after other nodes are updated.
  int* d_node_effected;
  gpuErrchk(cudaMalloc((void**) &d_node_effected, num_nodes * sizeof(int)));
  std::vector<int> node_effected_(num_nodes, 0);
  gpuErrchk(cudaMemcpy(d_node_effected, node_effected_.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
  // TODO: The list of effected nodes to update residuals. Dense approach.

  // Create h frontiers that will represent the splash.
  std::vector<int> edge_effected_(num_edges, 0);
  std::vector<int*> d_frontiers;
  for (int frontier_id = 0; frontier_id < h; ++frontier_id) {
    int* d_frontier;
    gpuErrchk(cudaMalloc((void**) &d_frontier, num_edges * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_frontier, edge_effected_.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    d_frontiers.push_back(d_frontier);
  }

  // We also want one final frontier that determines which edges need their residuals updated.
  int* d_edges_effected;
  gpuErrchk(cudaMalloc((void**) &d_edges_effected, num_edges * sizeof(int)));
  gpuErrchk(cudaMemcpy(d_edges_effected, edge_effected_.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
  // We will also need a workspace for this to avoid a race condition.
  int* d_edges_effected_workspace;
  gpuErrchk(cudaMalloc((void**) &d_edges_effected_workspace, num_edges * sizeof(int)));
  gpuErrchk(cudaMemcpy(d_edges_effected_workspace, edge_effected_.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));

  //
  // Setup GPU Runtime.
  //

  if (runtime_params.size() < 1) {
    std::cout << "RS requires parallelism divisor a, where p = 1/2^a." << std::endl;
    return std::tuple<float, std::vector<float>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>>(0.0,{},0,{},{});
  }
  float p = 1.0 / (float) std::pow(2, runtime_params[0]);

  // Determine grid/block sizes using CUDA Occupancy calculators.
  int minGridSize;

  int blockSizeSelect;
  int gridSizeSelect;
  int nodes_to_update = (int) (num_nodes * p);
  nodes_to_update = nodes_to_update == 0 ? 1 : nodes_to_update;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeSelect, rs_node_select, 0, 0));
  scale_launch_params(&gridSizeSelect, &blockSizeSelect, nodes_to_update);

  int blockSizeFrontier;
  int gridSizeFrontier;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeFrontier, rs_generate_next_frontier, 0, 0));
  scale_launch_params(&gridSizeFrontier, &blockSizeFrontier, num_edges);

  int blockSizeCalcUpdates;
  int gridSizeCalcUpdates;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeCalcUpdates, rs_calculate_updates, 0, 0));
  scale_launch_params(&gridSizeCalcUpdates, &blockSizeCalcUpdates, num_edges);

  int blockSizeEdgeResiduals;
  int gridSizeEdgeResiduals;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeEdgeResiduals, rs_compute_edge_residuals, 0, 0));
  scale_launch_params(&gridSizeEdgeResiduals, &blockSizeEdgeResiduals, num_edges);

  int blockSizeNodeResiduals;
  int gridSizeNodeResiduals;
  gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeNodeResiduals, rs_compute_node_residuals, 0, 0));
  scale_launch_params(&gridSizeNodeResiduals, &blockSizeNodeResiduals, num_nodes);

  // We instantiate here because we want to know the number of nodes we will be updating.
  std::vector<int> node_ids_out_;
  for (int i = 0; i < num_nodes; i += ceiling_division(num_nodes, nodes_to_update)) {
    node_ids_out_.push_back(i);
  }
  int i = 1;
  while(node_ids_out_.size() < num_nodes) {
    node_ids_out_.push_back(i);
    ++i;
    if (i % ceiling_division(num_nodes, nodes_to_update) == 0)
      ++i;
  }

  int* d_node_ids_out;
  gpuErrchk(cudaMalloc((void**) &d_node_ids_out, num_nodes * sizeof(int)));
  gpuErrchk(cudaMemcpy(d_node_ids_out, node_ids_out_.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));

  // Determine temporary device storage requirements for sorting.
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_node_residuals, d_node_residuals_out, d_node_ids, d_node_ids_out, num_nodes);

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

    // Start by generating our frontiers for this run.
    // First we select and mark the first frontier by selecting the top p nodes and adding their outgoing edges.
    rs_node_select<<<gridSizeSelect, blockSizeSelect>>>(infer_data.first, infer_data.second, d_node_ids_out, d_node_effected, d_frontiers[0], d_edges_effected, num_nodes, num_edges);

    // Now we generate the next h-1 frontiers.
    for (int frontier_id = 1; frontier_id < h; ++frontier_id) {
      rs_generate_next_frontier<<<gridSizeFrontier, blockSizeFrontier>>>(infer_data.first, infer_data.second, d_node_effected, d_frontiers[frontier_id - 1], d_frontiers[frontier_id], d_edges_effected, num_edges, true);
    }

    // Finally we need to extend our edges_effected by one more step.
    // To do this we need to use our workspace to avoid a race condition.
    // Start by copying the current edges effected into the workspace.
    gpuErrchk(cudaMemcpy(d_edges_effected_workspace, d_edges_effected, num_edges * sizeof(float), cudaMemcpyDeviceToDevice));

    // A little hacky this approach but it avoids a bunch of only slightly different code.
    rs_generate_next_frontier<<<gridSizeFrontier, blockSizeFrontier>>>(infer_data.first, infer_data.second, d_node_effected, d_edges_effected_workspace, d_edges_effected, d_edges_effected, num_edges, false);

    // We start with a backwards pass through our frontiers. This is the collection phase, where we make the root node aware of the leaves.
    for (int update = h; update > 0; --update) {
      // Operate on the specificed frontier.
      int* d_frontier = d_frontiers[update - 1];
      rs_calculate_updates<<<gridSizeCalcUpdates, blockSizeCalcUpdates>>>(infer_data.first, infer_data.second, d_frontier, num_edges, false);

      // Set the edges equal to the workspace each time.
      gpuErrchk(cudaMemcpy(infer_data.second.edges, infer_data.second.workspace, edge_rep_size * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Now we do the forwards pass through our frontiers. This is the distribution phase, where we make the leaves aware of the root.
    for (int update = 0; update < h; ++update) {
      // Operate on the specificed frontier.
      rs_calculate_updates<<<gridSizeCalcUpdates, blockSizeCalcUpdates>>>(infer_data.first, infer_data.second, d_frontiers[update], num_edges, true);

      // Set the edges equal to the workspace each time.
      gpuErrchk(cudaMemcpy(infer_data.second.edges, infer_data.second.workspace, edge_rep_size * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Compute the updated edge residuals.
    rs_compute_edge_residuals<<<gridSizeEdgeResiduals, blockSizeEdgeResiduals>>>(infer_data.first, infer_data.second, d_edge_residuals, d_edges_effected, num_edges);

    gpuErrchk(cudaMemcpy(infer_data.second.workspace, infer_data.second.edges, edge_rep_size * sizeof(float), cudaMemcpyDeviceToDevice));

    // Finally, compute the updated node residuals.
    rs_compute_node_residuals<<<gridSizeNodeResiduals, blockSizeNodeResiduals>>>(infer_data.first, infer_data.second, d_node_residuals, d_edge_residuals, d_node_effected, num_nodes);

    // Clear frontiers. Edges effected and nodes effected are already cleared.
    for (int frontier_id = 0; frontier_id < h; ++frontier_id) {
      gpuErrchk(cudaMemcpy(d_frontiers[frontier_id], edge_effected_.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Sort node_residuals using CUB device radix sort.
    // Run sorting operation
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_node_residuals, d_node_residuals_out, d_node_ids, d_node_ids_out, num_nodes);

    // Check largest residual. If it's less than epsilon, we know we've converged.
    gpuErrchk(cudaMemcpy(top_residual, d_node_residuals_out, sizeof(float), cudaMemcpyDeviceToHost)); // Only copy back the first value.
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
 
  gpuErrchk(cudaFree(d_temp_storage));
  gpuErrchk(cudaFree(d_node_residuals));
  gpuErrchk(cudaFree(d_node_residuals_out));
  gpuErrchk(cudaFree(d_node_ids));
  gpuErrchk(cudaFree(d_node_ids_out));
  gpuErrchk(cudaFree(d_edge_residuals));
  gpuErrchk(cudaFree(d_node_effected));
  for (int frontier_id = 0; frontier_id < h; ++frontier_id) {
    gpuErrchk(cudaFree(d_frontiers[frontier_id]));
  }
  gpuErrchk(cudaFree(d_edges_effected));
  gpuErrchk(cudaFree(d_edges_effected_workspace));

  free(top_residual);
  free_gpu_data(infer_data);

  return std::tuple<float, std::vector<float>, int, std::vector<std::pair<int,int>>, std::vector<std::pair<float, int>>>(converged ? milliseconds : -1.0, result, converged ? iterations : -1, {}, {});
}

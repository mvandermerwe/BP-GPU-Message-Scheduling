//
// Created by Mark Van der Merwe, Summer 2018
//

#include "inference_helpers.h"
#include "infer_data.h"
#include <boost/heap/fibonacci_heap.hpp>
#include <cub.cuh>

__host__ __device__ int ceiling_division(int x, int y) {
  return x/y + (x%y != 0);
}

// Use Euclidian distance as the metric to determine if the operations have converged.
__device__ double message_delta(double* edges, double* workspace, int edge_start) {
  int message_length = workspace[edge_start];

  double sqr_diff_sum = 0.0;
  for (int category = 0; category < message_length; ++category) {
    sqr_diff_sum += powf((workspace[edge_start + 1 + category] - edges[edge_start + 1 + category]), 2);
  }
  return sqrtf(sqr_diff_sum);
}

__device__ void compute_message(device_graph d_graph, device_pgm d_pgm, int edge_id) {
  // Determine where the message value is in the edges array.
  int edges_index = d_pgm.edge_idx_to_edges_idx[edge_id];
  // Determine the size of the message (i.e., the number of categories).
  int size_of_message = (int) d_pgm.edges[edges_index]; // S
    
  // First, get the indices of all the relevant messages for this one.
  int relevant_edges_start = d_graph.edge_idx_to_incoming_edges[edge_id];
  int relevant_edges_end = d_graph.edge_idx_to_incoming_edges[edge_id + 1];
    
  // Update the message.
    
  // Start by clearing the workspace.
  for (int message_idx = 0; message_idx < size_of_message; ++message_idx) {
    d_pgm.workspace[edges_index + 1 + message_idx] = 0; // S
  }

  // Determine the starting index for the edge factor and for the node factor for this message.
  int edge_factor_idx_start = d_pgm.edge_idx_to_edge_factors_idx[edge_id];
  int node_factor_idx_start = d_pgm.edge_idx_to_node_factors_idx[edge_id];

  // Determine if our source node is the first or second node argument of the factor function.
  // Whether we are the first arg determines how to index into the correct factor values.
  int first_arg = d_pgm.edge_factors[edge_factor_idx_start] == edge_id; // S
  int source_category_count;
  if (first_arg) {
    source_category_count = d_pgm.edge_factors[edge_factor_idx_start + 1]; // S
  } else {
    source_category_count = d_pgm.edge_factors[edge_factor_idx_start + 2]; // S
  }
  // Move the factor index start forward to where the actual factor values are.
  edge_factor_idx_start += 3;

  // Loop through each setting of the destination node.
  for (int setting = 0; setting < size_of_message; ++setting) {
    double value = 0.0;

    for (int source_setting = 0; source_setting < source_category_count; ++source_setting) {
      double partial_value = 0.0;

      // Find the edge factor value.
      int edge_factor_idx;
      if (first_arg) {
	edge_factor_idx = edge_factor_idx_start + (source_setting * size_of_message) + setting;
      } else {
	edge_factor_idx = edge_factor_idx_start + source_setting + (setting * source_category_count);
      }
      partial_value = d_pgm.edge_factors[edge_factor_idx]; // S

      // Find the node factor value.
      partial_value *= d_pgm.node_factors[node_factor_idx_start + 1 + source_setting]; // S

      // Multiply by incoming messages.
      for (int message_idx = relevant_edges_start; message_idx < relevant_edges_end; ++message_idx) {
	int message_rep_start = d_pgm.edge_idx_to_edges_idx[d_graph.edge_incoming_edges[message_idx]]; // S
	partial_value *= d_pgm.edges[message_rep_start + 1 + source_setting]; // S
      }
	
      // Now add the partial value.
      value += partial_value;
    }

    // Write the value to the corresponding location.
    d_pgm.workspace[edges_index + 1 + setting] = value; // S
  }

  // Normalize message by dividing each element by it's sum.
  double sum = 0.0;
  for (int message_idx = 0; message_idx < size_of_message; ++message_idx) {
    double contribution = d_pgm.workspace[edges_index + 1 + message_idx]; // S
    sum += contribution;
  }

  for (int message_idx = 0; message_idx < size_of_message; ++message_idx) {
    double final_value = d_pgm.workspace[edges_index + 1 + message_idx] / sum; // S
    d_pgm.workspace[edges_index + 1 + message_idx] = final_value;
  }
}

__device__ void compute_marginal(device_graph d_graph, device_pgm d_pgm, int* marginal_to_node_id, int* marginal_to_marginal_rep, double* marginals, int marginal) {
  int node = marginal_to_node_id[marginal];
  
  // Determine which edges lead to the given node.
  int node_edges_index = d_graph.node_idx_to_incoming_edges[node];
  int num_edges = d_graph.node_idx_to_incoming_edges[node+1] - node_edges_index;

  // Determine where final marginals should be written to.
  int marginal_rep_start = marginal_to_marginal_rep[marginal];
  int num_categories = marginals[marginal_rep_start];

  // Now we simply go and calculate the various values.
  for (int relevant_edge_idx = 0; relevant_edge_idx < num_edges; ++relevant_edge_idx) {
    // Go through each value in the relevant edge and multiply it in.
    for (int message_idx = 0; message_idx < num_categories; ++message_idx) {
      marginals[marginal_rep_start + 1 + message_idx] *= d_pgm.edges[d_pgm.edge_idx_to_edges_idx[d_graph.node_incoming_edges[node_edges_index + relevant_edge_idx]] + 1 + message_idx];
    }
  }

  // Normalize message by dividing each element by it's sum.
  double sum = 0.0;
  for (int message_idx = 0; message_idx < num_categories; ++message_idx) {
    sum += marginals[marginal_rep_start + 1 + message_idx];
  }
  for (int message_idx = 0; message_idx < num_categories; ++message_idx) {
    marginals[marginal_rep_start + 1 + message_idx] = marginals[marginal_rep_start + 1 + message_idx] / sum;
  }
}

__global__ void compute_marginals(device_graph d_graph, device_pgm d_pgm, int* marginal_to_node_id, int* marginal_to_marginal_rep, double* marginals, int node_count) {

  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  for (int marginal = idx; marginal < node_count; marginal += step) {
    compute_marginal(d_graph, d_pgm, marginal_to_node_id, marginal_to_marginal_rep, marginals, marginal);
  }
}

#define NUM_THREADS 512
#define NUM_BLOCKS 8

std::vector<double> compute_marginals(pgm* pgm, device_graph d_graph, device_pgm d_pgm, bool verbose) {
  // Start by sending the needed marginal data over to the GPU.
    int* d_marginalize_node_ids;
    gpuErrchk(cudaMalloc((void**) &d_marginalize_node_ids, pgm->marginalize_node_ids.size() * sizeof(int)));
    int* d_marginal_to_marginal_rep_idx;
    gpuErrchk(cudaMalloc((void**) &d_marginal_to_marginal_rep_idx, pgm->marginal_to_marginal_rep_idx.size() * sizeof(int)));
    double* d_marginal_rep;
    gpuErrchk(cudaMalloc((void**) &d_marginal_rep, pgm->marginal_rep.size() * sizeof(double)));
  
    // Send data over.
    gpuErrchk(cudaMemcpy(d_marginalize_node_ids, pgm->marginalize_node_ids.data(), pgm->marginalize_node_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_marginal_to_marginal_rep_idx, pgm->marginal_to_marginal_rep_idx.data(), pgm->marginal_to_marginal_rep_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_marginal_rep, pgm->marginal_rep.data(), pgm->marginal_rep.size() * sizeof(double), cudaMemcpyHostToDevice));

    // Compute Marginal_Rep Kernel Launch.
    compute_marginals<<<NUM_BLOCKS, NUM_THREADS>>>(d_graph, d_pgm, d_marginalize_node_ids, d_marginal_to_marginal_rep_idx, d_marginal_rep, pgm->marginalize_node_ids.size());

    // Now bring the marginal distributions back and print them out.
    double* h_marginal_rep = (double*) malloc(pgm->marginal_rep.size() * sizeof(double));
    gpuErrchk(cudaMemcpy(h_marginal_rep, d_marginal_rep, pgm->marginal_rep.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // Free up memory used for marginalization.
    gpuErrchk(cudaFree(d_marginalize_node_ids));
    gpuErrchk(cudaFree(d_marginal_to_marginal_rep_idx));
    gpuErrchk(cudaFree(d_marginal_rep));

    std::vector<double> result;
    result.assign(h_marginal_rep, h_marginal_rep + pgm->marginal_rep.size());
    free(h_marginal_rep);

    return result;
}

__global__ void filter_updates(int num_edges, double* residuals, int* need_update_filtered, double filter) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  // Iterate through all the edges assigned to this thread. This is likely as well balanced in terms of work as we can get.
  for (int edge_id = idx; edge_id < num_edges; edge_id += step) {
    need_update_filtered[edge_id] = residuals[edge_id] > filter ? 1 : 0;
  }
}

int edges_to_update(int num_edges, double* d_residuals, double cutoff) {
  // Intialize memory to perform reduction.
  int* d_edges_to_update;
  gpuErrchk(cudaMalloc(&d_edges_to_update, num_edges * sizeof(int)));
  int* d_edges_to_update_sum;
  gpuErrchk(cudaMalloc(&d_edges_to_update_sum, 1 * sizeof(int)));
  int h_edges_to_update_sum;

  filter_updates<<<NUM_THREADS, NUM_BLOCKS>>>(num_edges, d_residuals, d_edges_to_update, cutoff);

  // Now perform CUB reduction.
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_edges_to_update, d_edges_to_update_sum, num_edges);
  // Allocate temporary storage
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  // Run sum-reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_edges_to_update, d_edges_to_update_sum, num_edges);

  // Copy back result.
  gpuErrchk(cudaMemcpy(&h_edges_to_update_sum, d_edges_to_update_sum, sizeof(int), cudaMemcpyDeviceToHost));
  
  // Clean up.
  gpuErrchk(cudaFree(d_temp_storage));
  gpuErrchk(cudaFree(d_edges_to_update));
  gpuErrchk(cudaFree(d_edges_to_update_sum));

  return h_edges_to_update_sum;
}

__global__ void filter_updates(int num_edges, double* residuals, double epsilon, double* rand_vals, double p, int* need_update_filtered) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  // Iterate through all the edges assigned to this thread. This is likely as well balanced in terms of work as we can get.
  for (int edge_id = idx; edge_id < num_edges; edge_id += step) {
    need_update_filtered[edge_id] = residuals[edge_id] >= epsilon && rand_vals[edge_id] > p ? 1 : 0;
  }
}

int edges_to_update(int num_edges, double* d_residuals, double epsilon, double* d_rand_vals, double p) {
  // Intialize memory to perform reduction.
  int* d_edges_to_update;
  gpuErrchk(cudaMalloc(&d_edges_to_update, num_edges * sizeof(int)));
  int* d_edges_to_update_sum;
  gpuErrchk(cudaMalloc(&d_edges_to_update_sum, 1 * sizeof(int)));
  int h_edges_to_update_sum;

  filter_updates<<<NUM_THREADS, NUM_BLOCKS>>>(num_edges, d_residuals, epsilon, d_rand_vals, p, d_edges_to_update);

  // Now perform CUB reduction.
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_edges_to_update, d_edges_to_update_sum, num_edges);
  // Allocate temporary storage
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  // Run sum-reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_edges_to_update, d_edges_to_update_sum, num_edges);

  // Copy back result.
  gpuErrchk(cudaMemcpy(&h_edges_to_update_sum, d_edges_to_update_sum, sizeof(int), cudaMemcpyDeviceToHost));
  
  // Clean up.
  gpuErrchk(cudaFree(d_temp_storage));
  gpuErrchk(cudaFree(d_edges_to_update));
  gpuErrchk(cudaFree(d_edges_to_update_sum));

  return h_edges_to_update_sum;
}

__host__ __device__ void print_doubles(double* values, int count) {
  for (int value_idx = 0; value_idx < count; ++value_idx) {
    printf("%f, ", values[value_idx]);
  }
  printf("\n");
}

__host__ __device__ void print_ints(int* values, int count) {
  for (int value_idx = 0; value_idx < count; ++value_idx) {
    printf("%d, ", values[value_idx]);
  }
  printf("\n");
}

void print_gpu_data(int* d_data, int size) {
  int* h_data;
  h_data = (int *) malloc(size * sizeof(int));
  
  cudaMemcpy(h_data, d_data, size*sizeof(int), cudaMemcpyDeviceToHost);
  print_ints(h_data, size);
  free(h_data);
}

void print_gpu_data(double* d_data, int size) {
  double* h_data;
  h_data = (double *) malloc(size * sizeof(double));
  
  cudaMemcpy(h_data, d_data, size*sizeof(double), cudaMemcpyDeviceToHost);
  print_doubles(h_data, size);
  free(h_data);
}

// Generate the correct gridSize and blockSize for the given kernel and problem sizing.
void scale_launch_params(int* gridSize, int* blockSize, int problemSize) {
  if (problemSize < *blockSize) {
    *blockSize = problemSize;
    *gridSize = 1;
  } else {
    *gridSize = (problemSize + *blockSize - 1) / *blockSize;
  }
}

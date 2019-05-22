//
// Created by Mark Van der Merwe, Summer 2018
//

#include "infer_data.h"
#include "../header.h"

// TODO: We can use streams here to run a decent amount of this stuff in parallel.

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

std::pair<device_graph, device_pgm> setup_gpu_data(pgm* pgm) {

  //--------------------//
  //  Build the Graph   //
  //--------------------//

  // Node to incoming
  int* d_node_idx_to_incoming_edges;
  gpuErrchk(cudaMalloc((void**) &d_node_idx_to_incoming_edges, pgm->pgm_graph->node_idx_to_incoming_edges.size() * sizeof(int)));
  int* d_node_incoming_edges;
  gpuErrchk(cudaMalloc((void**) &d_node_incoming_edges, pgm->pgm_graph->node_incoming_edges.size() * sizeof(int)));

  // Node to outgoing
  int* d_node_idx_to_outgoing_edges;
  gpuErrchk(cudaMalloc((void**) &d_node_idx_to_outgoing_edges, pgm->pgm_graph->node_idx_to_outgoing_edges.size() * sizeof(int)));
  int* d_node_outgoing_edges;
  gpuErrchk(cudaMalloc((void**) &d_node_outgoing_edges, pgm->pgm_graph->node_outgoing_edges.size() * sizeof(int)));

  // Edge to incoming
  int* d_edge_idx_to_incoming_edges;
  gpuErrchk(cudaMalloc((void**) &d_edge_idx_to_incoming_edges, pgm->pgm_graph->edge_idx_to_incoming_edges.size() * sizeof(int)));
  int* d_edge_incoming_edges;
  gpuErrchk(cudaMalloc((void**) &d_edge_incoming_edges, pgm->pgm_graph->edge_incoming_edges.size() * sizeof(int)));

  // Edge to outgoing
  int* d_edge_idx_to_outgoing_edges;
  gpuErrchk(cudaMalloc((void**) &d_edge_idx_to_outgoing_edges, pgm->pgm_graph->edge_idx_to_outgoing_edges.size() * sizeof(int)));
  int* d_edge_outgoing_edges;
  gpuErrchk(cudaMalloc((void**) &d_edge_outgoing_edges, pgm->pgm_graph->edge_outgoing_edges.size() * sizeof(int)));

  // Edge to dest node idx
  int* d_edge_idx_to_dest_node_idx;
  gpuErrchk(cudaMalloc((void**) &d_edge_idx_to_dest_node_idx, pgm->pgm_graph->edge_idx_to_dest_node_idx.size() * sizeof(int)));

  // Copy over all values into memories we just allocated.
  gpuErrchk(cudaMemcpy(d_node_idx_to_incoming_edges, pgm->pgm_graph->node_idx_to_incoming_edges.data(), pgm->pgm_graph->node_idx_to_incoming_edges.size() * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_node_incoming_edges, pgm->pgm_graph->node_incoming_edges.data(), pgm->pgm_graph->node_incoming_edges.size() * sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_node_idx_to_outgoing_edges, pgm->pgm_graph->node_idx_to_outgoing_edges.data(), pgm->pgm_graph->node_idx_to_outgoing_edges.size() * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_node_outgoing_edges, pgm->pgm_graph->node_outgoing_edges.data(), pgm->pgm_graph->node_outgoing_edges.size() * sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_edge_idx_to_incoming_edges, pgm->pgm_graph->edge_idx_to_incoming_edges.data(), pgm->pgm_graph->edge_idx_to_incoming_edges.size() * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_edge_incoming_edges, pgm->pgm_graph->edge_incoming_edges.data(), pgm->pgm_graph->edge_incoming_edges.size() * sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_edge_idx_to_outgoing_edges, pgm->pgm_graph->edge_idx_to_outgoing_edges.data(), pgm->pgm_graph->edge_idx_to_outgoing_edges.size() * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_edge_outgoing_edges, pgm->pgm_graph->edge_outgoing_edges.data(), pgm->pgm_graph->edge_outgoing_edges.size() * sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_edge_idx_to_dest_node_idx, pgm->pgm_graph->edge_idx_to_dest_node_idx.data(), pgm->pgm_graph->edge_idx_to_dest_node_idx.size() * sizeof(int), cudaMemcpyHostToDevice));

  // Build the encapsulating Graph struct.
  device_graph d_graph;
  d_graph.node_idx_to_incoming_edges = d_node_idx_to_incoming_edges;
  d_graph.node_incoming_edges = d_node_incoming_edges;
  d_graph.node_idx_to_outgoing_edges = d_node_idx_to_outgoing_edges;
  d_graph.node_outgoing_edges = d_node_outgoing_edges;
  d_graph.edge_idx_to_incoming_edges = d_edge_idx_to_incoming_edges;
  d_graph.edge_incoming_edges = d_edge_incoming_edges;
  d_graph.edge_idx_to_outgoing_edges = d_edge_idx_to_outgoing_edges;
  d_graph.edge_outgoing_edges = d_edge_outgoing_edges;
  d_graph.edge_idx_to_dest_node_idx = d_edge_idx_to_dest_node_idx;

  //--------------------//
  //    Build the PGM   //
  //--------------------//

  // Edge representations:
  int* d_edge_idx_to_edges_idx;
  gpuErrchk(cudaMalloc((void**) &d_edge_idx_to_edges_idx, pgm->edge_idx_to_edges_idx.size() * sizeof(int)));
  double* d_edges;
  gpuErrchk(cudaMalloc((void**) &d_edges, pgm->edges.size() * sizeof(double)));
  double* d_workspace; // Workspace will be used to write the new values to each iteration.
  gpuErrchk(cudaMalloc((void**) &d_workspace, pgm->edges.size() * sizeof(double)));

  // Edge factor representations:
  int* d_edge_idx_to_edge_factors_idx;
  gpuErrchk(cudaMalloc((void**) &d_edge_idx_to_edge_factors_idx, pgm->edge_idx_to_edge_factors_idx.size() * sizeof(int)));
  double* d_edge_factors;
  gpuErrchk(cudaMalloc((void**) &d_edge_factors, pgm->edge_factors.size() * sizeof(double)));

  // Node factor representations:
  int* d_edge_idx_to_node_factors_idx;
  gpuErrchk(cudaMalloc((void**) &d_edge_idx_to_node_factors_idx, pgm->edge_idx_to_node_factors_idx.size() * sizeof(int)));
  double* d_node_factors;
  gpuErrchk(cudaMalloc((void**) &d_node_factors, pgm->node_factors.size() * sizeof(double)));

  // Now copy over all values into the memories we just allocated.
  gpuErrchk(cudaMemcpy(d_edge_idx_to_edges_idx, pgm->edge_idx_to_edges_idx.data(), pgm->edge_idx_to_edges_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_edges, pgm->edges.data(), pgm->edges.size() * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_workspace, pgm->edges.data(), pgm->edges.size() * sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_edge_idx_to_edge_factors_idx, pgm->edge_idx_to_edge_factors_idx.data(), pgm->edge_idx_to_edge_factors_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_edge_factors, pgm->edge_factors.data(), pgm->edge_factors.size() * sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_edge_idx_to_node_factors_idx, pgm->edge_idx_to_node_factors_idx.data(), pgm->edge_idx_to_node_factors_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_node_factors, pgm->node_factors.data(), pgm->node_factors.size() * sizeof(double), cudaMemcpyHostToDevice));

  // Built the encapsulating PGM struct.
  device_pgm d_pgm;
  d_pgm.edge_idx_to_edges_idx = d_edge_idx_to_edges_idx;
  d_pgm.edges = d_edges;
  d_pgm.workspace = d_workspace;
  d_pgm.edge_idx_to_edge_factors_idx = d_edge_idx_to_edge_factors_idx;
  d_pgm.edge_factors = d_edge_factors;
  d_pgm.edge_idx_to_node_factors_idx = d_edge_idx_to_node_factors_idx;
  d_pgm.node_factors = d_node_factors;
  
  return std::pair<device_graph, device_pgm>(d_graph, d_pgm);
}

void free_gpu_data(std::pair<device_graph, device_pgm> infer_data) {
  gpuErrchk(cudaFree(infer_data.first.node_idx_to_incoming_edges));
  gpuErrchk(cudaFree(infer_data.first.node_incoming_edges));
  gpuErrchk(cudaFree(infer_data.first.node_idx_to_outgoing_edges));
  gpuErrchk(cudaFree(infer_data.first.node_outgoing_edges));
  gpuErrchk(cudaFree(infer_data.first.edge_idx_to_incoming_edges));
  gpuErrchk(cudaFree(infer_data.first.edge_incoming_edges));
  gpuErrchk(cudaFree(infer_data.first.edge_idx_to_outgoing_edges));
  gpuErrchk(cudaFree(infer_data.first.edge_outgoing_edges));
  gpuErrchk(cudaFree(infer_data.second.edge_idx_to_edges_idx));
  gpuErrchk(cudaFree(infer_data.second.edges));
  gpuErrchk(cudaFree(infer_data.second.workspace));
  gpuErrchk(cudaFree(infer_data.second.edge_idx_to_edge_factors_idx));
  gpuErrchk(cudaFree(infer_data.second.edge_factors));
  gpuErrchk(cudaFree(infer_data.second.edge_idx_to_node_factors_idx));
  gpuErrchk(cudaFree(infer_data.second.node_factors));
}

std::pair<device_graph, device_pgm> setup_cpu_data(pgm* pgm) {

  // Build encapsulating Graph struct.
  device_graph d_graph;
  d_graph.node_idx_to_incoming_edges = pgm->pgm_graph->node_idx_to_incoming_edges.data();
  d_graph.node_incoming_edges = pgm->pgm_graph->node_incoming_edges.data();
  d_graph.node_idx_to_outgoing_edges = pgm->pgm_graph->node_idx_to_outgoing_edges.data();
  d_graph.node_outgoing_edges = pgm->pgm_graph->node_outgoing_edges.data();
  d_graph.edge_idx_to_incoming_edges = pgm->pgm_graph->edge_idx_to_incoming_edges.data();
  d_graph.edge_incoming_edges = pgm->pgm_graph->edge_incoming_edges.data();
  d_graph.edge_idx_to_outgoing_edges = pgm->pgm_graph->edge_idx_to_outgoing_edges.data();
  d_graph.edge_outgoing_edges = pgm->pgm_graph->edge_outgoing_edges.data();
  d_graph.edge_idx_to_dest_node_idx = pgm->pgm_graph->edge_idx_to_dest_node_idx.data();

  // Build encapsulating PGM struct.
  device_pgm d_pgm;
  d_pgm.edge_idx_to_edges_idx = pgm->edge_idx_to_edges_idx.data();
  d_pgm.edges = pgm->edges.data();
  d_pgm.workspace = (double *) malloc(pgm->edges.size() * sizeof(double));
  d_pgm.edge_idx_to_edge_factors_idx = pgm->edge_idx_to_edge_factors_idx.data();
  d_pgm.edge_factors = pgm->edge_factors.data();
  d_pgm.edge_idx_to_node_factors_idx = pgm->edge_idx_to_node_factors_idx.data();
  d_pgm.node_factors = pgm->node_factors.data();

  return std::pair<device_graph, device_pgm>(d_graph, d_pgm);
}

void free_cpu_data(std::pair<device_graph, device_pgm> infer_data) {
  free(infer_data.second.workspace);

  // Rest should be freed automatically thanks to deconstruction.
}

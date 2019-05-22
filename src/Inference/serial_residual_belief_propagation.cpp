//
// Created by Mark Van der Merwe, Fall 2018
//

#include "serial_inference_helpers.h"
#include <utility>
#include <vector>
#include "../header.h"
#include <ctime>
#include <iostream>
#include <chrono>
#include <boost/heap/fibonacci_heap.hpp>

struct residual_pair_ordering {
  bool operator() (const std::pair<int, double>& lhs, const std::pair<int, double>& rhs) const {
    return std::get<1>(lhs) < std::get<1>(rhs);
  }
};

std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, double epsilon, int timeout, std::vector<int> runtime_params, bool verbose) {
  
  int num_edges = pgm->edge_idx_to_edges_idx.size();
  int edge_size = pgm->edges.size();

  if (verbose) {
    std::cout << num_edges << std::endl;
  }

  std::vector<double> workspace = pgm->edges;
  bool converged = false;

  // Use priority queue to select which node to update next.
  // residual_priority_queue rpq(num_edges, 10.0);
  boost::heap::fibonacci_heap<std::pair<int, double>, boost::heap::compare<residual_pair_ordering>> rpq;

  // Init rpq with initial residual (10.0) and get handles.
  typedef typename boost::heap::fibonacci_heap<std::pair<int, double>, boost::heap::compare<residual_pair_ordering>>::handle_type handle_t;
  std::vector<handle_t> edge_handles;
  for (int i = 0; i < num_edges; ++i) {
    handle_t handle_i = rpq.push(std::pair<int, double>(i, 10.0));
    edge_handles.push_back(handle_i);
  }

  std::cout << "Starting inference." << std::endl;

  // Time our code.
  std::clock_t begin = std::clock();
  std::clock_t since;
  float time = 0.0;

  auto start = std::chrono::steady_clock::now();

  int iterations = 0;
  while(!converged && time < timeout) {
    ++iterations;
    
    // Get element with top residual and update that edge.
    int top_residual_element_id = std::get<0>(rpq.top());
    rpq.pop();
    // This writes back into the edges itself.
    double message_diff = compute_message(pgm, workspace, top_residual_element_id, true);
    // Right after an update, residual is zero, since no new information has been propagated yet.
    //rpq.update(edge_handles[top_residual_element_id], std::pair<int, double>(top_residual_element_id, 0.0));
    edge_handles[top_residual_element_id] = rpq.push(std::pair<int, double>(top_residual_element_id, 0.0));

    // Now recompute residuals for that edge and all effected edges (i.e. outgoing edges).
    int outgoing_edge_start = pgm->pgm_graph->edge_idx_to_outgoing_edges[top_residual_element_id];
    int outgoing_edge_end = pgm->pgm_graph->edge_idx_to_outgoing_edges[top_residual_element_id + 1];
    for (int outgoing_edge_idx = outgoing_edge_start; outgoing_edge_idx < outgoing_edge_end; ++outgoing_edge_idx) {
      int outgoing_edge_id = pgm->pgm_graph->edge_outgoing_edges[outgoing_edge_idx];
      // Update into workspace but don't transfer back (so not saved). This gets us the residual.
      double outgoing_edge_residual = compute_message(pgm, workspace, outgoing_edge_id, false);
      rpq.update(edge_handles[outgoing_edge_id], std::pair<int, double>(outgoing_edge_id, outgoing_edge_residual));
    }

    converged = std::get<1>(rpq.top()) < epsilon;

    since = std::clock();
    time = float(since - begin) / CLOCKS_PER_SEC;
  }

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  auto converge_time = std::chrono::duration<double, std::milli>(diff).count();
  std::cout << "Time: " << converge_time << " ms." << std::endl;

  //rpq.print();

  // Compute the final marginals.
  compute_marginals(pgm);

  if (verbose) {
    // Print results:
    print_doubles(pgm->marginal_rep);
    std::cout << "Iterations: " << iterations << std::endl;
  }

  std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> results(converged ? converge_time : -1.0, pgm->marginal_rep, converged ? iterations : -1, {}, {});
  return results;
}

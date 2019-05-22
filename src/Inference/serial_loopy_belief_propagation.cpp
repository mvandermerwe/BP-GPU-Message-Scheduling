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

std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, double epsilon, int timeout, std::vector<int> runtime_params, bool verbose) {
  
  int num_edges = pgm->edge_idx_to_edges_idx.size();
  int edge_size = pgm->edges.size();

  std::vector<double> workspace = pgm->edges;
  bool converged = false;

  std::cout << "Starting inference." << std::endl;

  // Time our code.
  std::clock_t begin = std::clock();
  std::clock_t since;
  float time = 0.0;

  auto start = std::chrono::steady_clock::now();

  int iterations = 0;
  while(!converged && time < timeout) {
    ++iterations;
    converged = true;

    for (int edge_id = 0; edge_id < num_edges; ++edge_id) {
      double delta = compute_message(pgm, workspace, edge_id);
      
      if (delta > epsilon) {
	converged = false;
      }
    }

    // Swap our edge pointer to point to our workspace, where the updates are currently written.
    pgm->edges.swap(workspace);

    since = std::clock();
    time = float(since - begin) / CLOCKS_PER_SEC;
  }

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  auto converge_time = std::chrono::duration<double, std::milli>(diff).count();
  std::cout << "Time: " << converge_time << " ms." << std::endl;

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

//
// Created by Mark Van der Merwe, Fall 2018
//

#include <vector>
#include <string>
#include <iostream>

#ifndef TIME_TESTS_HELPERS_H
#define TIME_TESTS_HELPERS_H

void create_folders(std::string folder_root);

void write_marginals_to_file(std::vector<double> marginals, std::string results_folder, std::string file);

void write_edge_convergence_to_file(std::vector<std::pair<int, int>> edge_convergence_iterations, std::vector<std::pair<float, int>> edge_convergence_time, std::string results_folder, std::string file);

void write_convergence_results_to_file(std::vector<std::string> run_files, std::vector<float> converge_runtime, std::vector<int> converge_iterations, std::vector<std::pair<float, float>> runtime_cumulative, std::vector<std::pair<int, float>> iterations_cumulative, std::string results_folder);

// Convert from convergent information (time, iterations) to cumulative percentage of converged runs.
template<class T>
std::vector<std::pair<T, float>> convert_to_cumulative(std::vector<T> converge_info, int n, float timeout_ms, int i) {
  std::sort(converge_info.begin(), converge_info.end());

  std::vector<std::pair<T, float>> result;

  result.push_back(std::pair<T, float>(0, 0));

  int converge_count = 0;

  for(T converge_time: converge_info) {
    if (converge_time < 0)
      continue;

    ++converge_count;

    result.push_back(std::pair<T, float>(converge_time, (float)converge_count / (float)n));
  }

  // Add at the end a final "asymptotic" point at timeout s w/ their max size.
  result.push_back(std::pair<T, float>(timeout_ms, (float)converge_count / (float)n));

  return result;
}

#endif // TIME_TESTS_HELPERS_H

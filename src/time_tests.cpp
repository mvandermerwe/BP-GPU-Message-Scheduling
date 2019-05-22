//
// Created by Mark Van der Merwe, Summer 2018
//

#include "header.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "Tests/time_tests_helpers.h"
#include <sstream>

// Given the upper and lower bounds of the arguments, create the cartesian product of the settings.
std::vector<std::vector<int>> setup_runtime_parameters(int num_args, std::vector<std::pair<int, int>> upper_lower_bounds) {
  // Create hyperparameters (vector of vectors).
  std::vector<std::vector<int>> runtime_parameters;

  // Set up intial hyperparameter with first settings.
  std::vector<int> initial_settings;
  for (int arg_num = 0; arg_num < num_args; ++arg_num) {
    initial_settings.push_back(std::get<0>(upper_lower_bounds[arg_num]));
  }
  runtime_parameters.push_back(initial_settings);

  while(true) {
    int arg_num = num_args - 1;
    while(true) {
      if (initial_settings[arg_num] + 1 > std::get<1>(upper_lower_bounds[arg_num])) {
	initial_settings[arg_num] = std::get<0>(upper_lower_bounds[arg_num]);
	--arg_num;
	if (arg_num == -1) {
	  break;
	}
      } else {
	initial_settings[arg_num] = initial_settings[arg_num] + 1;
	break;
      }
    }

    if (arg_num == -1)
      break;
    runtime_parameters.push_back(initial_settings);
  }

  return runtime_parameters;
}

// Generate a folder name for the given settings.
std::string settings_folder_name(std::string folder_prefix, std::vector<int> settings) {
  std::stringstream ss;
  ss << folder_prefix;
  ss << "R";
  for (int setting : settings) {
    ss << "_" << setting;
  }
  return ss.str();
}

int main(int argc, char*argv[]) {

  //
  // Parse inputs.
  //

  if (argc < 6) {
    std::cout << "Usage:\n   ./benchmark.out <prefix> <file_prefix> <n> <Results folder> <timeout (sec)> <num args> <...arg lower upper int pairs (in order expected)>" << std::endl;
    return 0;
  }

  // Number of times we repeat each run (for more accurate runtime).
  int total_runs = 1;

  std::string prefix = argv[1];
  std::string file_prefix = argv[2];
  int n = std::stoi(argv[3]);
  std::string results_folder = argv[4];
  float timeout = std::stof(argv[5]);
  float timeout_ms = timeout * 1000.;
  int num_args = std::stoi(argv[6]);
  
  if (argc != 7 + (num_args * 2)) {
    std::cout << "Number of args does not match. Should provide upper and lower int bounds for each argument." << std::endl;
    return 0;
  }
  
  std::vector<std::pair<int, int>> upper_lower_bounds;

  for (int arg_num = 0; arg_num < num_args; ++arg_num) {
    upper_lower_bounds.push_back(std::make_pair(std::stoi(argv[7 + (arg_num * 2)]), std::stoi(argv[7 + (arg_num * 2) + 1])));
  }

  std::vector<std::vector<int>> runtime_parameters;
  if (num_args > 0) {
    runtime_parameters = setup_runtime_parameters(num_args, upper_lower_bounds);
  } else {
    runtime_parameters.push_back({});
  }

  // Each converge info contains the runtime and the iterations until convergence for that run.
  std::vector<std::string> run_files;
  std::vector<float> converge_runtime;
  std::vector<int> converge_iterations;

  int i = 0;

  for(std::vector<int> runtime_params : runtime_parameters) {
    float p = 1.0;
    // Create a folder for this p.
    std::string results_folder_p = settings_folder_name(results_folder, runtime_params);
    create_folders(results_folder_p);

    for (int graph_num = 0; graph_num < n; ++graph_num) {
      std::vector<std::pair<float, int>> graph_runtimes;

      std::string filename = file_prefix + std::to_string(graph_num) + ".txt";
      std::string full_filename = prefix + filename;
      std::cout << "Starting: " << full_filename << std::endl;
      run_files.push_back(filename);

      pgm* test_pgm = new pgm(full_filename);
	
      for (int run = 0; run < total_runs; ++run) {
	std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> results = infer(test_pgm, 0.0001, timeout, runtime_params, false);
	graph_runtimes.push_back(std::pair<float,int>(std::get<0>(results), std::get<2>(results)));
	write_marginals_to_file(std::get<1>(results), results_folder_p, filename);
	write_edge_convergence_to_file(std::get<3>(results), std::get<4>(results), results_folder_p, filename);
      }

      free(test_pgm);

      // Average values.
      float sum_runtime = 0.;
      int sum_iterations = 0;
      for (std::pair<float, int> runtime_info: graph_runtimes) {
	sum_runtime += runtime_info.first;
	sum_iterations += runtime_info.second;
      }
      float avg_runtime = sum_runtime / total_runs;
      int avg_iterations = sum_iterations / total_runs;

      converge_runtime.push_back(avg_runtime);
      converge_iterations.push_back(avg_iterations);
    }

    // At each p, build the cumulative convergence plot.
    std::vector<std::pair<float, float>> runtime_cumulative = convert_to_cumulative<float>(converge_runtime, n, timeout_ms, i);
    std::vector<std::pair<int, float>> iterations_cumulative = convert_to_cumulative<int>(converge_iterations, n, timeout_ms, i);

    write_convergence_results_to_file(run_files, converge_runtime, converge_iterations, runtime_cumulative, iterations_cumulative, results_folder_p);

    converge_runtime.clear();
    converge_iterations.clear();
    run_files.clear();

    ++i;
  }

  return 0;
}

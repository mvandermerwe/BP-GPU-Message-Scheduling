//
// Created by Mark Van der Merwe, Fall 2018
//

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "time_tests_helpers.h"

void create_folders(std::string folder_root) {
  boost::filesystem::path dest_folder = folder_root;
  boost::filesystem::create_directory(dest_folder);

  // Create the marginals folder.
  std::string marginal_folder_name = folder_root + "/Marginals";
  boost::filesystem::path marginal_folder = marginal_folder_name;
  boost::filesystem::create_directory(marginal_folder);

  // Create edge convergence folders (iterations + time).
  std::string edge_convergence_folder_name = folder_root + "/EdgeConvergence";
  boost::filesystem::path edge_convergence_folder = edge_convergence_folder_name;
  boost::filesystem::create_directory(edge_convergence_folder);

  boost::filesystem::path edge_convergence_folder_iterations = edge_convergence_folder_name + "/Iterations";
  boost::filesystem::create_directory(edge_convergence_folder_iterations);
  boost::filesystem::path edge_convergence_folder_time = edge_convergence_folder_name + "/Time";
  boost::filesystem::create_directory(edge_convergence_folder_time);
}

void write_marginals_to_file(std::vector<double> marginals, std::string results_folder, std::string file) {
  std::ofstream marginals_file;
  marginals_file.open(results_folder + "/Marginals/" + file);
  
  for (double marginal: marginals) {
    marginals_file << marginal << ", ";
  }

  marginals_file.close();
}

void write_edge_convergence_to_file(std::vector<std::pair<int, int>> edge_convergence_iterations, std::vector<std::pair<float, int>> edge_convergence_time, std::string results_folder, std::string file) {
  std::ofstream edge_convergence_file;

  // Write iterations.
  edge_convergence_file.open(results_folder + "/EdgeConvergence/Iterations/" + file);
  
  for (std::pair<int, int> iteration_count : edge_convergence_iterations) {
    edge_convergence_file << std::get<0>(iteration_count) << "," << std::get<1>(iteration_count) << std::endl;
  }

  edge_convergence_file.close();

  // Write time.
  edge_convergence_file.open(results_folder + "/EdgeConvergence/Time/" + file);
  
  for (std::pair<float, int> time_count : edge_convergence_time) {
    edge_convergence_file << std::get<0>(time_count) << "," << std::get<1>(time_count) << std::endl;
  }

  edge_convergence_file.close();
  
}

void write_convergence_results_to_file(std::vector<std::string> run_files, std::vector<float> converge_runtime, std::vector<int> converge_iterations, std::vector<std::pair<float, float>> runtime_cumulative, std::vector<std::pair<int, float>> iterations_cumulative, std::string results_folder) {
  std::ofstream converge_runtimes_file;
  converge_runtimes_file.open(results_folder + "/runtimes.txt");
  std::ofstream converge_iterations_file;
  converge_iterations_file.open(results_folder + "/iterations.txt");
  std::ofstream runtime_cumulative_file;
  runtime_cumulative_file.open(results_folder + "/runtime_cumulative.txt");
  std::ofstream iterations_cumulative_file;
  iterations_cumulative_file.open(results_folder + "/iterations_cumulative.txt");

  // Start by writing converge runtimes and iterations.
  for (int i = 0; i < run_files.size(); ++i) {
    std::string run_file = run_files[i];

    converge_runtimes_file << run_file << "," << converge_runtime[i] << "\n";
    converge_iterations_file << run_file << "," << converge_iterations[i] << "\n";
  }

  // Now write the cumulative convergence results using runtime and iterations.
  for (int i = 0; i < runtime_cumulative.size(); ++i) {
    std::pair<float, float> runtime_cumulative_point = runtime_cumulative[i];
    std::pair<int, float> iterations_cumulative_point = iterations_cumulative[i];

    runtime_cumulative_file << runtime_cumulative_point.first << "," << runtime_cumulative_point.second << "\n";
    iterations_cumulative_file << iterations_cumulative_point.first << "," << iterations_cumulative_point.second << "\n";
  }

  converge_runtimes_file.close();
  converge_iterations_file.close();
  runtime_cumulative_file.close();
  iterations_cumulative_file.close();
}


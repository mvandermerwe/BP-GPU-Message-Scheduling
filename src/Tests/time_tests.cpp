//
// Created by Mark Van der Merwe, Summer 2018
//

#include "header.h"
#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char*argv[]) {

  if (argc != 2) {
    std::cout << "Usage:\n   ./ising_benchmark.out <verbose>\n   Verbose should be 1 or 0 depending upon whether you want output from runs or not." << std::endl;
    return 0;
  }

  // Timeout limit on a single run.
  float timeout = 10.0;
  // Verbosity.
  int verbose = std::stoi(argv[1]);

  // Run timing tests on the ising models.
  int lower = 20;
  int upper = 40;
  int increment = 10;
  int n = 10;
  std::string prefix = "../../benchmarks/ising-models/";

  std::map<int, std::vector<float>> size_to_converge_times;

  for (int size = lower; size <= upper; size += increment) {
    for (int graph_num = 0; graph_num < n; ++graph_num) {
      std::string filename = prefix + "ising_" + std::to_string(size) + "_" + std::to_string(graph_num) + ".txt";
    
      pgm* ising_pgm = new pgm(filename);

      float time = infer(ising_pgm, 0.0001, timeout, verbose);
      size_to_converge_times[size].push_back(time);

      std::cout << size << ", " << time << std::endl;
    }
  }

  // Convert from convergent times to cumulative percentage of converged runs.
  for (int size = lower; size <= upper; size += increment) {
    std::vector<float> converge_times = size_to_converge_times[size];
    std::sort(converge_times.begin(), converge_times.end());

    std::cout << "Size: " << size << std::endl;
    int converge_count = 0;

    for(float converge_time: converge_times) {
      if (converge_time == -1)
	continue;

      ++converge_count;
      
      std::cout << converge_time << ", " << (float)converge_count / (float)n << std::endl;
    }
  }

  return 0;
}

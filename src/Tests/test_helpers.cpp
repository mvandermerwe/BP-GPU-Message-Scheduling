//
// Created by Mark Van der Merwe, Summer 2018
//

#include "test_helpers.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

void print_test_assert(bool result, std::string testname) {
  std::string result_string = result ? "PASS" : "FAIL";
  std::cout << "Test: " << testname << " - " << result_string << std::endl;
}

double avg_diff_arrays(std::vector<double> results, std::vector<double> gold) {
  // Array sizes should match.
  if (results.size() != gold.size()) {
    std::cout << "Bad size array." << std::endl;
    return -1;
  }

  // Average all differences.
  double sum = 0.0;
  for (int index = 0; index < results.size(); ++index) {
    sum += std::abs(results[index] - gold[index]);
  }

  return sum / ((results.size() / 3) * 2);
}

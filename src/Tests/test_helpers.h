//
// Created by Mark Van der Merwe, Summer 2018
//

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

template <class T>
void print_array(std::vector<T> data) {
  std::cout << data.size() << std::endl;
  for (T val: data) {
    std::cout << val << ", ";
  }
  std::cout << std::endl;
}

template <class T>
bool equal_arrays(std::vector<T> results, std::vector<T> gold) {
  // Array sizes should match.
  if (results.size() != gold.size()) {
    std::cout << "Bad size array." << std::endl;
    return false;
  }

  // Check each corresponding value.
  for (int index = 0; index < results.size(); ++index) {
    if (results[index] != gold[index]) {
      std::cout << "Fail on index " << index << ", expected " << gold[index] << " but found " << results[index] << "." << std::endl;
    }
  }

  // If we reach here, all good.
  return true;
}

template <class T>
bool equal_arrays(std::vector<T> results, std::vector<T> gold, T accuracy) {
  // Array sizes should match.
  if (results.size() != gold.size()) {
    std::cout << "Bad size array." << std::endl;
    return false;
  }

  bool equal = true;

  // Check each corresponding value.
  for (int index = 0; index < results.size(); ++index) {
    if (std::abs(results[index] - gold[index]) > accuracy) {
      std::cout << "Fail on index " << index << ", expected " << gold[index] << " but found " << results[index] << "." << std::endl;
      equal = false;
    }
  }

  return equal;
}

double avg_diff_arrays(std::vector<double> results, std::vector<double> gold);

void print_test_assert(bool result, std::string testname);

#endif // TEST_HELPERS_H

//
// Created by Mark Van der Merwe, Summer 2018
//

#ifndef CUPGM_PARALLEL_INFERENCE_INFER_H_
#define CUPGM_PARALLEL_INFERENCE_INFER_H_

#include <string>
#include <vector>
#include <tuple>
#include "../Representation/pgm.h"

// Perform inference to identify the posterior marginal distributions for the provided list of variables, to a level of convergence
// defined by the provided epsilon.
// Return the time elapsed until convergence, the result vector, the number of iterations, and the
// (iteration/time, edges not converged) vectors.
std::tuple<float, std::vector<double>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, double epsilon, int timeout, std::vector<int> runtime_params={}, bool verbose=false);

#endif // CUPGM_PARALLEL_INFERENCE_INFER_H_

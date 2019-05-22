//
// Created by Mark Van der Merwe, Fall 2018
//

#ifndef CUPGM_PARALLEL_INFERENCE_SERIAL_INFERENCE_HELPERS_H_
#define CUPGM_PARALLEL_INFERENCE_SERIAL_INFERENCE_HELPERS_H_

#include <vector>
#include "../header.h"
#include <iostream>

// Update given edge, return the message delta.
double compute_message(pgm* pgm, std::vector<double> &workspace, int edge_id, bool write_to_edges=false);

double message_delta(std::vector<double> edges, std::vector<double> workspace, int edge_start);

void compute_marginals(pgm* pgm);

void print_doubles(std::vector<double> values);

#endif // CUPGM_PARALLEL_INFERENCE_SERIAL_INFERENCE_HELPERS_H_

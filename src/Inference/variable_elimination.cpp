//
// Created by Mark Van der Merwe, Fall 2018.
//

#include "infer.h"

#include <tuple>
#include <utility>
#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <chrono>

// Only works for Ising Grids!

// Get the tau value for a given setting.
// Start by building a query index, then returning the value at that index.
float query_tau(std::map<int,int> settings, std::vector<float> &tau) {
  // Now for each node, the value for that node is multiplied by the
  // number of settings of the following nodes to add to the query index.
  int query_index = 0;
  int settings_of_prev = 1;

  for (int node_idx = tau[0]; node_idx >= 1; --node_idx) {
    // Get node value.
    int node = tau[node_idx];

    // Determine setting.
    int setting = settings[node];

    // Add to query by setting * previous settings.
    query_index += setting * settings_of_prev;

    // Now multiply number of settings to previous settings count.
    settings_of_prev *= tau[node_idx + tau[0]];
  }

  return tau[(2 * tau[0]) + 1 + query_index];
}

// Query the given edge potential using the settings.
float query_edge_potential(std::map<int, int> &settings, std::vector<float> &edge_factors, int edge_factor_start, std::vector<int> &edge_id_to_dest_node) {

  // First need to find out which nodes we are working with.
  int first_node = edge_id_to_dest_node[edge_factors[edge_factor_start]];
  int first_node_size = edge_factors[edge_factor_start + 1];
  int second_node = edge_id_to_dest_node[edge_factors[edge_factor_start] + 1];
  int second_node_size = edge_factors[edge_factor_start + 2];

  // Now build the query index based on the above.
  int query_index = (settings[first_node] * second_node_size) + settings[second_node];
  return edge_factors[edge_factor_start + 3 + query_index];
}

// For the given settings map, determine the value after summing out a given node.
float get_value(pgm* pgm, int node_id, std::vector<float> &node_unary_potential, std::map<int, int> &settings, std::vector<int> &edge_potential_starts, std::vector<float> &tau) {
  float value = 0.0;

  for (int node_setting = 0; node_setting < node_unary_potential.size(); ++node_setting) {
    float setting_value = 1.0;
    
    // Add the node to remove's setting to the map.
    settings[node_id] = node_setting;

    // Now for every relevant edge potential, send the settings and multiply!
    for (int edge_potential_start: edge_potential_starts) {
      setting_value *= query_edge_potential(settings, pgm->edge_factors, edge_potential_start, pgm->pgm_graph->edge_idx_to_dest_node_idx);
    }

    // Multiply by tau value (if necessary).
    if (tau.size() != 0) {
      setting_value *= query_tau(settings, tau);
    }

    // Multiply by unary potential as well.
    setting_value *= node_unary_potential[node_setting];

    value += setting_value;
  }

  return value;
}

// Increment the settings map. Return false if a further increment goes beyond the valid settings.
bool increment_settings(std::map<int, int> &settings, std::set<int> nodes, std::map<int, int> node_sizes) {

  for (std::set<int>::reverse_iterator node_iter = nodes.rbegin(); node_iter != nodes.rend(); ++node_iter) {
    int node = *node_iter;
    int current_setting = settings[node];

    if (current_setting < node_sizes[node] - 1) {
      settings[node] = current_setting + 1;
      return true;
    } else {
      settings[node] = 0;

      // If the node is the first node, it means we've gone through all settings.
      if (*node_iter == *nodes.begin()) {
	return false;
      }
    }
  }

  return false;
}

// Given a node to sum out and the previous tau (resid. factor), determine new
// tau summing out the given node.
// Tau encoding: # nodes, node ids, node sizes, {factor}
std::vector<float> sum_out(pgm* pgm, int node_id, std::vector<float> &tau, std::set<int> &summed_out) {
  std::vector<float> new_tau;
  std::map<int, int> new_tau_node_sizes;

  // Track where the edge potentials start for all relevant edges.
  std::vector<int> edge_potential_starts;

  // Get list of nodes in new tau:
  std::set<int> new_tau_nodes;
  if (tau.size() > 0) {
    for (int idx = 1; idx < 1 + tau[0]; ++idx) {
      if (tau[idx] != node_id) {
	new_tau_nodes.insert(tau[idx]);
	new_tau_node_sizes[tau[idx]] = tau[tau[0] + idx];
      }
    }
  }

  // Determine new nodes from new edges:
  for (std::pair<int, int> edge_factor: pgm->node_idx_to_edge_factor_idx[node_id]) {
    int other_node = std::get<0>(edge_factor);
    int edge_factor_start = std::get<1>(edge_factor);

    // If other node has been removed, we can ignore it, it'll be in the tau.
    if (summed_out.find(other_node) == summed_out.end()) {
      new_tau_nodes.insert(other_node);

      // Determine size of this new node.
      if (other_node == pgm->pgm_graph->edge_idx_to_dest_node_idx[pgm->edge_factors[edge_factor_start]]) {
	new_tau_node_sizes[other_node] = pgm->edge_factors[edge_factor_start + 2];
      } else {
	new_tau_node_sizes[other_node] = pgm->edge_factors[edge_factor_start + 1];
      }

      // Add its potential start to list of relevant potential starts.
      edge_potential_starts.push_back(edge_factor_start);
    }
  }

  // Write number of nodes.
  new_tau.push_back(new_tau_nodes.size());

  // Write tau nodes and sizes:
  // Also, create a map from node id to setting, starting at 0.
  std::map<int, int> settings;
  for (float node_id: new_tau_nodes) {
    new_tau.push_back(node_id);
    settings[node_id] = 0;
  }
  for (float node_id: new_tau_nodes) {
    new_tau.push_back(new_tau_node_sizes[node_id]);
  }
  int factor_start = new_tau.size();

  // Create vector holding unary potential for node to remove:
  int node_factor_start = pgm->node_idx_to_node_factor_idx[node_id];
  int node_size = pgm->node_factors[node_factor_start];
  std::vector<float> unary_potential(node_size);
  
  for (int node_setting = 0; node_setting < node_size; ++node_setting) {
    unary_potential[node_setting] = pgm->node_factors[node_factor_start + 1 + node_setting];
  }

  // Increment through all settings and determine tau values.
  new_tau.push_back(get_value(pgm, node_id, unary_potential, settings, edge_potential_starts, tau));
  while(increment_settings(settings, new_tau_nodes, new_tau_node_sizes)) {
    new_tau.push_back(get_value(pgm, node_id, unary_potential, settings, edge_potential_starts, tau));
  }
  return new_tau;
}

void print_tau(std::vector<float> tau) {
 for (int tau_idx = 0; tau_idx < tau.size(); ++tau_idx) {
    std::cout << tau[tau_idx] << ",";
  }
  std::cout << std::endl;
}

std::tuple<float, std::vector<float>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> infer(pgm* pgm, float epsilon, int timeout, std::vector<int> runtime_params, bool verbose) {

  // First pass: only calculating the last marginal.

  // Ordering for elimination: front to end.

  int num_edges = pgm->edge_idx_to_edges_idx.size();
  int num_nodes = pgm->pgm_graph->node_idx_to_outgoing_edges.size()-1;

  std::vector<float> marginals;

  auto start = std::chrono::steady_clock::now();

  // Tau is a "residual" factor, after summing out some node.
  for (int node = 0; node < num_nodes; ++node) {
    std::vector<float> tau = {};

    std::set<int> summed_out;
    for (int node_id = 0; node_id < num_nodes; ++node_id) {
      if (node_id != node) {
        tau = sum_out(pgm, node_id, tau, summed_out);
	summed_out.insert(node_id);
      }
    }

    // Now multiply by the final unary potential:
    int node_unary_potential_start = pgm->node_idx_to_node_factor_idx[node];
    for (int tau_idx = 3; tau_idx < tau.size(); ++tau_idx) {
      tau[tau_idx] = tau[tau_idx] * pgm->node_factors[node_unary_potential_start + 1 + (tau_idx - 3)];
    }

    // Normalize:
    float total = 0.0;
    for (int tau_idx = 3; tau_idx < tau.size(); ++tau_idx) {
      total += tau[tau_idx];
    }
    for (int tau_idx = 3; tau_idx < tau.size(); ++tau_idx) {
      tau[tau_idx] = tau[tau_idx] / total;
    }

    marginals.push_back(2.0);
    for (int tau_idx = 3; tau_idx < tau.size(); ++tau_idx) {
      marginals.push_back(tau[tau_idx]);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;
  auto converge_time = std::chrono::duration<double, std::milli>(diff).count();

  std::tuple<float, std::vector<float>, int, std::vector<std::pair<int, int>>, std::vector<std::pair<float, int>>> results(converge_time, marginals, 0, {}, {});
  return results;
}

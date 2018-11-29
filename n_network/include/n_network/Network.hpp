#ifndef NETWORK_HPP
#define NETWORK_HPP 

#include "n_network/Layer.hpp"
#include <Eigen/Eigen>

template <typename L1, typename ... Ls>
struct Network {
  using start = Layer<L1, Ls ...>;
};


#endif // NETWORK_HPP

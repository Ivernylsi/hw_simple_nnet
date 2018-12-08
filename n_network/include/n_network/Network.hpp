#ifndef NETWORK_HPP
#define NETWORK_HPP 

#include "n_network/Layer.hpp"
#include "n_network/Loss.hpp"
#include <Eigen/Eigen>

template <typename L1, typename ... Ls>
struct Network {
  using start = Layer<L1, Ls ...>;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    return L.forward(in);
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    return L.backward(in);
  }

  private:
  start L;
};


#endif // NETWORK_HPP

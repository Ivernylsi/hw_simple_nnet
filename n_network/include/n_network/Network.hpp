#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "n_network/Layer.hpp"
#include "n_network/Loss.hpp"
#include <Eigen/Eigen>
#include <iostream>

template <typename L1, typename... Ls> struct Network {
  using start = Layer<L1, Ls...>;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) { return L.forward(in); }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) { return L.backward(in); }

  template <typename Loss = loss::MSE>
  double training(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y,
                const int iter = 100) {
    Loss loss;
    double a;
    for (int i = 0; i < iter; ++i) {
      auto f = forward(X);

      a = loss.forward(f, y);
     // if(i % 100 == 0)
//      printf("Loss on iteration %d : %f\n", i, a);
      auto l = loss.backward(y);
      backward(l);
    }
    return a;
  }

private:
  start L;
};

#endif // NETWORK_HPP

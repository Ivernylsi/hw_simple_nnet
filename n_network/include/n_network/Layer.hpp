#ifndef LAYER_HPP
#define LAYER_HPP

#include "n_network/Activations.hpp"
#include "n_network/Layers.hpp"

template <typename L1, typename... Ls> struct Layer {
  using curr = L1;
  using next = Layer<Ls...>;

  static_assert(curr::OutN == next::curr::InN,
                "OutPut of layer and input of next should be the same");

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    return NextL.forward(in);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    return NextL.backward(in);
  }

private:
  curr CurrL;
  next NextL;
};

template <typename L1> struct Layer<L1> {
  using curr = L1;
  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) { return in; }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) { return in.transpose(); }

private:
  curr Currl;
};

#endif // LAYER_HPP

#ifndef LAYERS_HPP
#define LAYERS_HPP
#include <Eigen/Eigen>
#include <iostream>
#include <random>

namespace {
double getNormal(const double &) {
  std::normal_distribution<double> runif(-1.0, 1.0);
  std::mt19937 rng;
  return runif(rng);
}
} // namespace

template <size_t N> struct Input {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) { return in; }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) { return in; }
};

template <size_t N, size_t K> struct Dense {
  static const size_t InN = N;
  static const size_t OutN = K;

  Dense() {
    auto kek = [](auto &a) { return getNormal(a);};
    w = Eigen::MatrixXd(InN, OutN).unaryExpr(kek);
  }

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    input = in;
    return in * w;
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    Eigen::MatrixXd grad = input.transpose() * in;
    input.resize(0, 0);
    w -= 0.1 * grad;
    return grad;
  }

private:
  Eigen::MatrixXd input;
  Eigen::MatrixXd w;
};

template <size_t N> struct Output {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) { return in; }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    return in * Eigen::VectorXd::Ones(InN);
  }
};

template <size_t N> struct DropOut {
  static const size_t InN = N;
  static const size_t OutN = N;
  static inline bool stop = false;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    if (stop)
      return in;
    d = Eigen::MatrixXd::Zero(InN, 0).unaryExpr(getUniform);
    return in * d.asDiagonal();
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    return in * d.asDiagonal();
  }

private:
  Eigen::VectorXd d;
  static double getUniform(const double &) {
    std::uniform_real_distribution<double> runif(0, 1.0);
    std::mt19937 rng;
    double rand = runif(rng);
    return (rand > 0.8 ? 1 : 0);
  }
};

#endif // LAYERS_HPP

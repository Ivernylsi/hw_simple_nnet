#ifndef LAYERS_HPP
#define LAYERS_HPP
#include <Eigen/Eigen>
#include <iostream>
#include <random>

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
  std::normal_distribution<double> runif(0.0, 0.1);
  std::mt19937 rng;

    w = Eigen::MatrixXd(InN, OutN);
    for(int i = 0; i < InN; ++i) 
      for(int j = 0; j < OutN; ++j)
        w(i,j) = runif(rng);
    b = runif(rng);
  }

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    input = in;
    return (in * w).array() + b;
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    Eigen::MatrixXd grad = input.transpose() * in;

    w -= 0.01 * grad;
    b -= 0.002 * in.sum();
    return  in * w.transpose();
  }

private:
  Eigen::MatrixXd input;
  Eigen::MatrixXd w;
  double b;
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
    d = Eigen::MatrixXd::Zero(InN, OutN).unaryExpr(getUniform);
    return in.array() * d.array();
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    return in.array() * d.array();
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

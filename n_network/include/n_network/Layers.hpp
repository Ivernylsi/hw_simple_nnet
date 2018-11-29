#ifndef LAYERS_HPP
#define LAYERS_HPP
#include <Eigen/Eigen>
#include <random>

template <size_t N> struct Input {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) { return in; }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}
};

template <size_t N, size_t K> struct Dense {
  static const size_t InN = N;
  static const size_t OutN = K;

  Dense() { w.setZero(); }

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) { return in * w; }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}

private:
  Eigen::Matrix<double, InN, OutN> w;
};

template <size_t N> struct Output {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) { return in; }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}
};

template <size_t N> struct DropOut {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) { 
    Eigen::VectorXd d = Eigen::MatrixXd::Zero(InN, 0).unaryExpr(getUniform);
    return in * d.asDiagonal(); }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}

private:
  static double getUniform(const double &) {
    std::uniform_real_distribution<double> runif(0, 1.0);
    std::mt19937 rng;
    double rand = runif(rng);
    return (rand > 0.8 ? 1 : 0);
  }
};

#endif // LAYERS_HPP

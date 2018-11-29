#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP
#include <Eigen/Eigen>

template <size_t N> struct Relu {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto relu = [](const auto &el) { return std::max(el, 0.0); };
    return in.unaryExpr(relu);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}
};

template <size_t N> struct LeakyRelu {
  static const size_t InN = N;
  static const size_t OutN = N;

  const double alpha = 0.01;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto leakyrelu = [&](const auto &el) {
      return (el >= 0 ? el : alpha * el);
    };
    return in.unaryExpr(leakyrelu);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}
};

template <size_t N> struct SoftPlus {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto softplus = [](const auto &el) { return std::log(1 + std::exp(el)); };
    return in.unaryExpr(softplus);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}
};

template <size_t N> struct Sigmoid {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto sigmoid = [](const auto &el) { return 1 / (1 + std::exp(-el)); };
    return in.unaryExpr(sigmoid);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}
};

template <size_t N> struct SoftMax {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto (*fexp)(double)->double = std::exp;
    decltype(in) expon = in.unaryExpr(fexp);
    return expon / expon.sum();
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}
};

template <size_t N> struct LogSoftMax {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto (*fexp)(double)->double = std::exp;
    auto (*flog)(double)->double = std::log;

    decltype(in) expon = in.unaryExpr(fexp);
    decltype(in) out = expon / expon.sum();

    return out.unaryExpr(flog);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {}
};

#endif // ACTIVATIONS_HPP

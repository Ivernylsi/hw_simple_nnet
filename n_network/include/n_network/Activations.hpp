#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP
#include <Eigen/Eigen>
#include <iostream>

struct ActivationBase {
  Eigen::MatrixXd input;
};

template <size_t N> struct Relu : ActivationBase {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto relu = [](const double &el) { return std::max(el, 0.0); };
    input = in;
    return in.unaryExpr(relu);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    auto relu = [](const auto &el) { return (el > 0 ? 1.0 : 0.0); };
    input = input.unaryExpr(relu);
    return input.array() * in.array();
  }
};

template <size_t N> struct LeakyRelu : ActivationBase {
  static const size_t InN = N;
  static const size_t OutN = N;

  const double alpha = 0.001;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto leakyrelu = [&](const auto &el) {
      return (el > 0 ? el : alpha * el);
    };
    input = in;
    return in.unaryExpr(leakyrelu);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    auto leakyrelu = [&](const auto &el) { return (el > 0 ? 1 : alpha); };
    input = input.unaryExpr(leakyrelu);
    return in * input;
  }
};

template <size_t N> struct SoftPlus : ActivationBase {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto softplus = [](const auto &el) { return std::log(1 + std::exp(el)); };
    return in.unaryExpr(softplus);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    auto softplus = [](const auto &el) { return exp(el) / (1 + exp(el)); };
    return input.unaryExpr(softplus) * in;
  }
};

template <size_t N> struct Sigmoid : ActivationBase {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto sigmoid = [](const auto &el) { return 1 / (1 + std::exp(-el)); };
    input = in;
    return in.unaryExpr(sigmoid);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    auto temp = forward(input);
    return (temp.array() / (1 - temp.array())) * in.array();
  }
};

template <size_t N> struct SoftMax : ActivationBase {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto (*fexp)(double)->double = std::exp;
    input = in;
    decltype(in) expon = in.unaryExpr(fexp);
    return expon / expon.sum();
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    auto fexp = [](const auto &a) { return std::exp(a);};
    Eigen::MatrixXd expon = input.unaryExpr(fexp);
    expon /= expon.sum();
    auto deriv = [&](int i, int j) {
      if (i == j)
        return expon(i) * (1 - expon(i));
      else
        return -expon(i) * expon(j);
    };
    Eigen::MatrixXd out = input;
    for (int i = 0; i < out.rows(); ++i)
      for (int j = 0; j < out.cols(); ++j) {
        out(i, j) = deriv(i, j);
      }
    return out * in;
  }
};

template <size_t N> struct LogSoftMax : ActivationBase {
  static const size_t InN = N;
  static const size_t OutN = N;

  Eigen::MatrixXd forward(const Eigen::MatrixXd &in) {
    auto (*fexp)(double)->double = std::exp;
    auto (*flog)(double)->double = std::log;
    input = in;
    decltype(in) expon = in.unaryExpr(fexp);
    decltype(in) out = expon / expon.sum();

    return out.unaryExpr(flog);
  }
  Eigen::MatrixXd backward(const Eigen::MatrixXd &in) {
    auto (*fexp)(double)->double = std::exp;
    Eigen::MatrixXd expon = input.unaryExpr(fexp);
    expon /= expon.sum();
    auto deriv = [&](int i, int j) {
      if (i == j)
        return expon(i) * (1 - expon(i));
      else
        return -expon(i) * expon(j);
    };
    Eigen::MatrixXd out(InN, OutN);
    for (int i = 0; i < out.rows(); ++i)
      for (int j = 0; j < out.cols(); ++j) {
        out(i, j) = deriv(i, j);
      }
    Eigen::MatrixXd log = input;
    auto (*flog)(double)->double = std::log;
    input = input.unaryExpr(flog);
    return input * out * in;
  }
};

#endif // ACTIVATIONS_HPP

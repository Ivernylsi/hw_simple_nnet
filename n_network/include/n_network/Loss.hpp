#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Eigen>

namespace loss {
struct MSE {

  double forward(const Eigen::MatrixXd &in, const Eigen::MatrixXd &y) {
    return (in.array() - y.array()).abs2().mean();
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in,
                           const Eigen::MatrixXd &y) {}
};

struct CrossEntropy {
  double forward(const Eigen::MatrixXd &in, const Eigen::MatrixXd &y) {
    auto logref = [](const auto &y) { return std::log(1 + y); };
    auto ans = in;
    ans = ans.unaryExpr(logref);
    return -(ans.array() * y.array()).sum();
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &in,
                           const Eigen::MatrixXd &y) {}
};

} // namespace loss
#endif // LOSS_HPP

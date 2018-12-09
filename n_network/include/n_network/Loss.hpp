#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Eigen>

namespace loss {
struct MSE {

  double forward(const Eigen::MatrixXd &in, const Eigen::MatrixXd &y) {
    input = in;
    return (in.array() - y.array()).abs2().sum()/y.rows();
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &y) {
    Eigen::MatrixXd kek = (input.array() - y.array()) / y.rows();
    return kek;
  }
private:
  Eigen::MatrixXd input;
};

struct CrossEntropy {
  double forward(const Eigen::MatrixXd &in, const Eigen::MatrixXd &y) {
    input = in;
    auto logref = [](const auto &y) { return std::log(y); };
    auto ans = in;
    ans = ans.unaryExpr(logref);
    return -(ans.array() * y.array()).sum();
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &y) {
    auto logref = [](const auto &y) { return 1/(y); };
    Eigen::MatrixXd ans = input.unaryExpr(logref);
    ans = -(ans.array() * y.array());
    return ans;
  }

private:
  Eigen::MatrixXd input;
};

struct CrossEntropySoftMax {
  double forward(const Eigen::MatrixXd &in, const Eigen::MatrixXd &y) {
    //softmax part
    auto fexp = [](const auto &a) { return std::exp(a);};
    input = in;
    Eigen::MatrixXd expon = input.unaryExpr(fexp);
    for(int i = 0; i < expon.rows(); ++i)
    expon.row(i) /= expon.row(i).sum();

    // crossentropy part
    auto logref = [](const auto &a) { return std::log(a); };
    auto ans = expon;
    ans = ans.unaryExpr(logref);
    return -(ans.array() * y.array()).sum();
  }

  Eigen::MatrixXd backward(const Eigen::MatrixXd &y) {
    Eigen::MatrixXd kek = (input.array() - y.array()) / y.rows();
    return kek;
  }

private:
  Eigen::MatrixXd input;

};



} // namespace loss
#endif // LOSS_HPP

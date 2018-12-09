#include "dataset.h"
#include <iostream>
#include <n_network/Network.hpp>
#include <vector>

const int ENCODED_SIZE = 784;

// clang-format off
/*
using mynet = Network<Input<784>, 
                      Dense<784, 300>, 
                      Relu<300>, 
                      Dense<300, 90>,
                      Relu<90>, 
                      Dense<90, 10>, 
                      Output<10>>;
                      */
using mynet = Network<Input<784>,
                      Dense<784, 10>,
                      LeakyRelu<10>,
                      Output<10>>;
// clang-format on

Eigen::MatrixXd getLabels() {
  return read_Mnist_Label(
      "/home/little/ML/Nnetwork/data/train-labels-idx1-ubyte");
}

Eigen::MatrixXd getTrainData() {
  return (read_Mnist("/home/little/ML/Nnetwork/data/train-images-idx3-ubyte") /
          255.0);
}

auto interpretAns(const Eigen::MatrixXd &v) {
  double max = -100;
  int idx = 0;
  for (int i = 0; i < v.cols(); ++i)
    if (max <= v(0, i)) {
      max = v(i);
      idx = i;
    }
  auto ans = v;
  ans.setZero();
  ans(idx) = 1;
  return ans;
}

int main() {

  mynet net;
  auto y = getLabels();
  auto X = getTrainData();
  std::cout << "Dataset reading completed.\n";

  const int size = 1000;
  double ans;
  for (int k = 0; k < 100; ++k) {
    for (int i = 0; i < 10; ++i) {

      ans = net.training<loss::CrossEntropySoftMax>(
          X.block(size * i, 0, size * (i + 1), 784),
          y.block(size * i, 0, size * (i + 1), 10), 1);
    }
    std::cout << k << " Loss os " << ans << std::endl;
  }
  return 0;
}

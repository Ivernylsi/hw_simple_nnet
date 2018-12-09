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
                      Output<10>>;
// clang-format on

Eigen::MatrixXd getLabels() {
  return read_Mnist_Label(
      "/home/little/ML/Nnetwork/data/train-labels-idx1-ubyte");
}

Eigen::MatrixXd getTrainData() {
  return read_Mnist("/home/little/ML/Nnetwork/data/train-images-idx3-ubyte");
}

int main() {

  mynet net;
  auto y = getLabels();
  auto X = getTrainData();
  std::cout << "Dataset reading completed.\n";

  const int size = 100;
  net.training(X.block(0, 0, size, 784), y.block(0, 0, size, 10), 10);
  return 0;
}

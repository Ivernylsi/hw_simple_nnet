#include "dataset.h"
#include <iostream>
#include <n_network/Network.hpp>
#include <vector>

//clang-format off
template struct Network<Input<10>, Relu<10>, Dense<10, 20>, Dense<20, 20>,
                        Output<20>>;

using mynet =
    Network<Input<10>, Relu<10>, Dense<10, 20>, Dense<20, 20>, Output<20>>;
//clang-format on

Eigen::MatrixXd getLabels() {
  return read_Mnist_Label(
      "/home/little/ML/Nnetwork/data/train-labels-idx1-ubyte");
}

Eigen::MatrixXd getTrainData() {
  return read_Mnist("/home/little/ML/Nnetwork/data/train-images-idx3-ubyte");
}

int main() {

  mynet net;
  Eigen::MatrixXd a(1, 10);
  a.setOnes();

  std::cout << " forward\n" << net.forward(a) << std::endl;

  std::cout << " backward\n" << net.backward(a) << std::endl;

  LeakyRelu<10> r;
  Eigen::VectorXd aa(6);
  aa << -1, 1, 1, 0, -5, 10;
  std::cout << r.forward(aa) << std::endl;

  auto y = getLabels();
  auto X = getTrainData();
  std::cout << X.row(0) << std::endl;
  return 0;
}

#include <iostream>
#include <n_network/Network.hpp>

template struct Network<Input<10>, Dense<10, 20>, Dense<20, 20>, Output<20>>;
using mynet = Network<Input<10>, Dense<10, 20>, Dense<20, 20>, Output<20>>;

int main() {
  mynet net;
  Eigen::Vector3d a(1, 1, 1);
  std::cout << " forward\n" << net.forward(a) << std::endl;

  std::cout << " backward\n" << net.backward(a) << std::endl;

  LeakyRelu<10> r;
  Eigen::VectorXd aa(6);
  aa << -1, 1, 1, 0, -5, 10;
  std::cout << r.forward(aa) << std::endl;

  return 0;
}

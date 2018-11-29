#include <n_network/Network.hpp>
#include <iostream>

template struct 
        Network<Input<10>,
                Dense<10, 20>, 
                Dense<20, 20>, 
                Output<20>
                >;
using mynet = 
        Network<Input<10>,
                Dense<10, 20>, 
                Dense<20, 20>, 
                Output<20>
                >;

int main() { 
  
  std::cout << mynet::start::curr::InN << std::endl;  
  
  return 0; 
}

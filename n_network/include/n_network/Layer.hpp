#ifndef LAYER_HPP
#define LAYER_HPP 

#include "n_network/Layers.hpp"
#include "n_network/Activations.hpp"

template <typename L1, typename ... Ls>
struct Layer {
  using curr = L1;
  using next = Layer<Ls ...>;
};

template <typename L1>
struct Layer<L1> {
  using curr = L1;
};



#endif // LAYER_HPP

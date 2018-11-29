#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Eigen/Eigen>

template <size_t N> struct Input {
  static const size_t InN = N;
  static const size_t OutN = N;
};

template <size_t N, size_t K> struct Dense {
  static const size_t InN = N;
  static const size_t OutN = K;
};

template <size_t N> struct Output {
  static const size_t InN = N;
  static const size_t OutN = N;
};

template <size_t N> struct DropOut {
  static const size_t InN = N;
  static const size_t OutN = N;
};

#endif // LAYERS_HPP

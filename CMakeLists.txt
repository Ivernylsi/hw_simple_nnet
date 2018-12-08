project(neural_network)

cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -march=native -O3 -Wall -Wextra -std=c++17")

find_package(Eigen3 REQUIRED)

add_subdirectory(n_network)

add_executable(net_test
               main.cpp
               )

target_link_libraries(net_test n_network)

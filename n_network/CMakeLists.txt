add_library(n_network INTERFACE
           )


target_include_directories(n_network INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(n_network INTERFACE Eigen3::Eigen)

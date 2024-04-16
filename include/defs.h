//
// Created by davidl09 on 4/13/24.
//

#ifndef PERCEPTRON_DEFS_H
#define PERCEPTRON_DEFS_H

#define EIGEN_USE_THREADS
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <nlohmann/json.hpp>

using namespace Eigen;
namespace ranges = std::ranges;
using namespace nlohmann;
namespace fs = std::filesystem;
namespace views = std::ranges::views;

template <typename T>
concept Scalar = std::is_floating_point<T>::value;

enum class ACTIVATION {
    SIGMOID,
    RELU,
    TANH,
    NONE,
};


#endif //PERCEPTRON_DEFS_H

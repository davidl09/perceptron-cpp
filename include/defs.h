//
// Created by davidl09 on 4/13/24.
//

#ifndef PERCEPTRON_DEFS_H
#define PERCEPTRON_DEFS_H

using namespace Eigen;
namespace ranges = std::ranges;
using namespace nlohmann;
namespace fs = std::filesystem;

template <typename T>
concept Scalar = std::is_floating_point<T>::value;

enum class ACTIVATION {
    SIGMOID,
    RELU,
    TANH,
    NONE,
};


#endif //PERCEPTRON_DEFS_H

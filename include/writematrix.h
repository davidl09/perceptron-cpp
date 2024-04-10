//
// Created by davidl09 on 4/9/24.
//

#ifndef PERCEPTRON_WRITEMATRIX_H
#define PERCEPTRON_WRITEMATRIX_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;
using namespace Eigen;

/*
 * file layout:
 *      4 bytes -> number of columns
 *      4 bytes -> number of rows
 *      4 bytes -> size of data field in bytes
 *      sizeof(T) * cols * rows bytes -> data
 */

template <typename T> requires std::is_floating_point_v<T>
void writeMatrix(const MatrixX<T>& mat, const fs::path& path) {
    if (std::ofstream file{path, std::ios::out | std::ios::binary}) {
        const uint32_t rows = mat.rows(), cols = mat.cols(), size = mat.size() * sizeof(T);

        file.write(reinterpret_cast<const char *>(&cols), 4);
        file.write(reinterpret_cast<const char *>(&rows), 4);
        file.write(reinterpret_cast<const char *>(&size), 4);
        file.write((char *)(mat.data()), mat.size() * sizeof (T));
    }
    else throw std::runtime_error("Could not open file " + path.string());
}

template <typename T> requires std::is_floating_point_v<T>
void writeMatrix(const VectorX<T>& mat, const fs::path& path) {
    if (std::ofstream file{path, std::ios::out | std::ios::binary}) {
        const uint32_t rows = mat.rows(), cols = mat.cols(), size = mat.size() * sizeof(T);

        file.write(reinterpret_cast<const char *>(&cols), 4);
        file.write(reinterpret_cast<const char *>(&rows), 4);
        file.write(reinterpret_cast<const char *>(&size), 4);
        file.write((char *)(mat.data()), mat.size() * sizeof (T));
    }
    else throw std::runtime_error("Could not open file " + path.string());
}

template <typename T> requires std::is_floating_point_v<T>
MatrixX<T> readMatrix(const fs::path& path) {
    if (std::ifstream file{path, std::ios::binary}) {
        uint32_t cols, rows, size;
        file.read(reinterpret_cast<char *>(&cols), 4);
        file.read(reinterpret_cast<char *>(&rows), 4);
        file.read(reinterpret_cast<char *>(&size), 4);

        assert(sizeof(T) * rows * cols == size);
#ifndef NDEBUG
        std::cout << "Read Matrix with size " << size << ", cols " << cols << ", rows " << rows << ", type size " << size / (rows * cols) << '\n';
#endif

        MatrixX<T> result(rows, cols);
        file.read((char *)result.data(), size);
        return result;
    }
    else throw std::runtime_error("Could not open file " + path.string());
}

template <typename T> requires std::is_floating_point_v<T>
VectorX<T> readVector(const fs::path& path) {
    if (std::ifstream file{path, std::ios::binary}) {
        uint32_t cols, rows, size;
        file.read(reinterpret_cast<char *>(&cols), 4);
        file.read(reinterpret_cast<char *>(&rows), 4);
        file.read(reinterpret_cast<char *>(&size), 4);

        assert(sizeof(T) * rows * cols == size);
#ifndef NDEBUG
        std::cout << "Read Matrix with size " << size << ", cols " << cols << ", rows " << rows << ", type size " << size / (rows * cols) << '\n';
#endif

        VectorX<T> result(rows, cols);
        file.read((char *)result.data(), size);
        return result;
    }
    else throw std::runtime_error("Could not open file " + path.string());
}

#endif //PERCEPTRON_WRITEMATRIX_H

//
// Created by davidl09 on 4/14/24.
//

#ifndef PERCEPTRON_TRAINDATA_H
#define PERCEPTRON_TRAINDATA_H

#include "defs.h"
#include <filesystem>
#include "Perceptron.h"

template <Scalar T>
class TrainData {
public:
    TrainData(const fs::path& trainCSV, size_t inputSize, size_t outputSize)
    : inputSize(inputSize), outputSize(outputSize) {
        if (std::ifstream file{trainCSV}) {
            std::string line{};
            std::getline(file, line);
            while (std::getline(file, line)) {
                auto view = line | views::split(',');
                std::vector<std::string> data;
                for (const auto& str : view) {
                    data.emplace_back(str.begin(), str.end());
                }
                assert(data.size() == 2);
                inputData.push_back(std::stod(data[0]));
                outputData.push_back(std::stod(data[1]));
            }
        }
        else throw std::invalid_argument("Could not open data file " + trainCSV.string());
    }

    std::pair<VectorX<T>, VectorX<T>> getIoPair() {
        auto gen = std::make_unique<std::mt19937>(std::random_device{}());
        std::uniform_int_distribution<ssize_t> dist{0, inputData.size() - static_cast<decltype(inputData.size())>(std::max(inputSize, outputSize))};
        auto index = dist(*gen);
        return std::make_pair(
                Map<VectorX<T>>{&(inputData[index]), inputSize},
                Map<VectorX<T>>{&(outputData[index + inputSize - outputSize]), outputSize}
        );
    }

    std::pair<VectorX<T>, VectorX<T>> getIoPair(size_t index) {
        if (index > inputData.size() - static_cast<decltype(inputData.size())>(std::max(inputSize, outputSize)))
            throw std::out_of_range("Requested index not in dataset");
        return std::make_pair(
                Map<VectorX<T>>{&(inputData[index]), inputSize},
                Map<VectorX<T>>{&(outputData[index + inputSize - outputSize]), outputSize}
        );
    }

private:
    std::vector<T> inputData, outputData;
    const size_t inputSize, outputSize;
};


#endif //PERCEPTRON_TRAINDATA_H

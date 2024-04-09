//
// Created by davidl09 on 4/9/24.
//

#ifndef PERCEPTRON_TRAININGPARAMS_H
#define PERCEPTRON_TRAININGPARAMS_H

#include <nlohmann/json.hpp>
#include <fstream>

namespace fs = std::filesystem;
using namespace nlohmann;

class TrainingParams {
public:
    explicit TrainingParams(const fs::path& jsonFilePath) {
        json data;
        if (std::ifstream file{jsonFilePath}) {
            std::string fileContents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            try {
                data = json::parse(fileContents);

                epochs = data["epochs"].template get<size_t>();
                batchSize = data["batchSize"].template get<size_t>();
                threshhold = data["threshold"].template get<double>();
                learningRate = data["learningRate"].template get<double>();

            }
            catch (json::exception& e) {
                throw std::invalid_argument(std::string{"Error encountered while parsing training parameters: "} + e.what());
            }
        }
        else throw std::invalid_argument("File " + jsonFilePath.string() + " could not be found.\n");
    }

    size_t getEpochs() const {
        return epochs;
    }

    size_t getBatchSize() const {
        return batchSize;
    }

    double getThreshHold() const {
        return threshhold;
    }

    double getLearningRate() const {
        return learningRate;
    }

private:
    size_t epochs, batchSize;
    double threshhold, learningRate;
};


#endif //PERCEPTRON_TRAININGPARAMS_H

//
// Created by davidl09 on 4/7/24.
//

#ifndef PERCEPTRON_PERCEPTRON_H
#define PERCEPTRON_PERCEPTRON_H

#include "defs.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>
#include "writematrix.h"
#include "TrainingParams.h"


template <Scalar T>
class Perceptron {
public:

    std::vector<T> predict(std::vector<T> input) {
        VectorX<T> result = Map<VectorX<T>, Unaligned>(input.data(), input.size());

        for (auto& layer : layers) {
            result = layer.propagate(result);
        }

        return {result.begin(), result.end()};
    }

    double updateWeights(VectorX<T>&& input, VectorX<T>&& targetOut, T learningRate = 1e-5) {
        if (input.size() != this->numInputs()) {
            throw std::invalid_argument("Input dimensionality " + std::to_string(input.size()) + " does not match model input dimensions " + std::to_string(this->inputSize));
        }
        if (targetOut.size() != this->numOutputs()) {
            throw std::invalid_argument("Input dimensionality " + std::to_string(input.size()) + " does not match model input dimensions " + std::to_string(this->inputSize));
        }

        for (auto& layer : layers) {
            input = layer.propagate(input);
        }

        auto& output = input;

        VectorX<T> outputError = targetOut - output;
        const double error =  0.5 * pow(outputError.array(), 2).sum();

        for (auto layer = layers.rbegin(); layer < layers.rend(); ++layer) {
            VectorX<T> delta = outputError.cwiseProduct(layer->activation_dx(layer->output)); // Element-wise multiplication

            layer->bias += learningRate * delta; // Update bias
            layer->weights += learningRate * delta * layer->input.transpose(); // Weight change; // Update weights

            outputError = (layer->weights.transpose() * delta).eval(); // Error for the previous layer
        }

        return error;

    }

    double updateWeights(const std::vector<T>& data, const std::vector<T>& target, T learningRate = 1e-5) {
        VectorX<T>
                input = Map<VectorX<T>, Unaligned>(const_cast<T *>(data.data()), data.size()),
                targetOut = Map<VectorX<T>, Unaligned>(const_cast<T *>(target.data()), target.size());

        return updateWeights(std::move(input), std::move(targetOut), learningRate);
    }

    static json readAndValidateModelJson(const fs::path& filename) {
        json data;
        if (std::ifstream file(filename); file) {
            std::string file_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            try {
                data = json::parse(file_contents);
            }
            catch (std::exception& e) {
                throw std::runtime_error("Could not parse file " + filename.string() + ": " + e.what());
            }
        } else throw std::runtime_error("Could not find file " + filename.string());

        for (const auto& elem : {"inputs", "layers"}) {
            if (not data.contains(elem)) {
                throw std::invalid_argument(std::string{"Missing json attribute "} + elem + '\n');
            }
        }

        for (const auto& elem : data["layers"]) {
            if (not elem.contains("activation") or not elem.contains("size")) {
                throw std::invalid_argument(std::string{"Missing json attribute "} + std::string{elem} + '\n');
            }
        }

        return data;
    }

    static Perceptron newFromJson(const fs::path& path) {
        return fromJson(readAndValidateModelJson(path));
    }

    void saveToFolder(const fs::path& folderName) {
        if (not fs::exists(folderName)) {
            fs::create_directory(folderName);
        }
        auto jsonPath = folderName / "model.json";

        if (std::ofstream modelFile{jsonPath}) {
            json jsonData{};
            jsonData["inputs"] = this->numInputs();
            jsonData["layers"] = json::array();

            for (const auto& layer : layers) {
                jsonData["layers"].push_back({{"activation", layer.getStrActivationFunction()},{"size", layer.size()}});
            }
            modelFile << jsonData;
        } else throw std::runtime_error("Could not create file " + jsonPath.string());

        for (auto layer = layers.begin(); layer < layers.end(); ++layer) {
            writeMatrix(layer->weights, folderName / fs::path("weights" + std::to_string(layer - layers.begin()) + ".bin"));
            writeMatrix(layer->bias, folderName / fs::path("bias" + std::to_string(layer - layers.begin()) + ".bin"));
        }
    }

    static Perceptron<T> readFromFolder(const fs::path& path) {
        if (not fs::exists(path)) {
            throw std::invalid_argument("Could not find folder " + path.string());
        }

        const json data = readAndValidateModelJson(path / "model.json");

        Perceptron<T> result{data.at("inputs").template get<size_t>()};

        const json& layers = data.at("layers");

        for (auto layer = layers.begin(); layer < layers.end(); ++layer) {
            result.addLayer(
                    layer->at("size").template get<size_t>(),
                            parseActivationFuncStr(layer->at("activation").template get<std::string>()),
                            readMatrix<T>(path / fs::path("weights" + std::to_string(layer - layers.begin()) + ".bin")),
                    readMatrix<T>(path / fs::path("bias" + std::to_string(layer - layers.begin()) + ".bin"))
            );
        }

        return result;
    }

    friend std::ostream& operator<<(std::ostream& stream, const Perceptron& p) {
        stream << "Layers: " << p.layers.size() << '\n';

        for (const auto& layer : p.layers) {
            stream << layer << '\n';
        }
        return stream;
    }

    size_t numInputs() {
        return inputSize;
    }

    size_t numOutputs() {
        return layers.back().weights.rows();
    }

private:

    explicit Perceptron(const std::vector<size_t>& shape, bool lastLayerHasActivation = false, std::vector<ACTIVATION> layerFuncs = {}) {
        srand(
                []() -> unsigned {
                    std::mt19937 gen{std::random_device{}()};
                    std::uniform_int_distribution<unsigned> dist;
                    return dist(gen);
                }()
        ); //seed random gen for eigen random();

        //for a network with N layers including input/output, layers_ has length N - 1, and shape has length N.
        if (shape.size() < 3) {
            throw std::invalid_argument(
                    "Network must have at least one hidden layer, has "
                    + std::to_string(shape.size())
                    + " total layers\n"
            );
        }

        inputSize = shape.front();

        layerFuncs.resize(shape.size() - 1, ACTIVATION::TANH);
        if (not lastLayerHasActivation) {
            layerFuncs.back() = ACTIVATION::NONE;
        }

        for (auto i = shape.begin() + 1; i < shape.end(); ++i) {
            layers.push_back(std::move(PerceptronLayer(*i, i[-1], layerFuncs[i - shape.begin() - 1])));
        }
        assert(layers.size() == layerFuncs.size());
    }

    explicit Perceptron(size_t inputShape)
    : inputSize(inputShape) {}

    void addLayer(size_t width, ACTIVATION func, MatrixX<T>&& weights, VectorX<T>&& bias) {
        const auto prevLayerSize = (layers.empty() ? inputSize : layers.back().size());
        layers.push_back(PerceptronLayer{width, prevLayerSize, func});
    }

    static Perceptron fromJson(const json& data) {
        std::vector<ACTIVATION> activations;
        std::vector<size_t> shape;
        shape.push_back(data["inputs"]);

        for (const auto& elem : data["layers"]) {
            shape.push_back(elem["size"].template get<size_t>());
            activations.push_back(parseActivationFuncStr(elem["activation"].template get<std::string>()));
        }
        return Perceptron{shape, true, activations};
    }

    struct PerceptronLayer {
    public:
        PerceptronLayer(size_t layerSize, size_t prevLayerSize, ACTIVATION activation = ACTIVATION::SIGMOID)
                : weights(layerSize, prevLayerSize), input(prevLayerSize), output(layerSize), bias(layerSize), activationFuncID(activation)
        {
            switch (activation) {
                case ACTIVATION::RELU:
                    this->activation = relu;
                    this->activation_dx = relu_dx;
                    break;

                case ACTIVATION::SIGMOID:
                    this->activation = sigmoid;
                    this->activation_dx = sigmoid_dx;
                    break;

                case ACTIVATION::TANH:
                    this->activation = tanh;
                    this->activation_dx = tanh_dx;
                    break;

                case ACTIVATION::NONE:
                    this->activation = [](const VectorX<T>& in) -> VectorX<T> {
                        return in;
                    };
                    this->activation_dx = [](const VectorX<T>& in) -> VectorX<T> {
                        return VectorX<T>::Ones(in.size());
                    };
            }

            weights.setRandom();
            bias.setRandom();
        }

        const VectorX<T>& propagate(const VectorX<T>& in) {
            this->input = in;
            return this->output = activation(weights * in + bias);
        }

        [[nodiscard]] size_t size() const {
            return bias.size();
        }

        friend std::ostream& operator<<(std::ostream& stream, const PerceptronLayer& layer) {
            stream << layer.weights << '\n';
            return stream;
        }

        constexpr auto getStrActivationFunction() const {
            switch (this->activationFuncID) {
                case ACTIVATION::SIGMOID:
                    return "sigmoid";
                case ACTIVATION::RELU:
                    return "relu";
                case ACTIVATION::TANH:
                    return "tanh";
                case ACTIVATION::NONE:
                    return "none";
            }
            return "";
        };

        static VectorX<T> sigmoid(const VectorX<T>& input) {
            return VectorX<T>::Ones(input.size()).array() / (VectorX<T>::Ones(input.size()).array() + exp((-input).array())).array();
        }

        static VectorX<T> sigmoid_dx(const VectorX<T>& input) {
            return sigmoid(input) * (VectorX<T>::Ones(input.size()) - sigmoid(input));
        }

        static VectorX<T> tanh(const VectorX<T>& input) {
            return Eigen::tanh(input.array());
        }

        static VectorX<T> tanh_dx(const VectorX<T>& input) {
            return VectorX<T>::Ones(input.size()).array() - pow(input.array().tanh(), 2);
        }

        static VectorX<T> relu(const VectorX<T>& input) {
            VectorX<T> result = input;
            ranges::for_each(result, [](auto& elem) {
                elem = (elem > 0 ? elem : 0);
            });
            return result;
        }

        static VectorX<T> relu_dx(const VectorX<T>& input) {
            VectorX<T> result = input;
            ranges::for_each(result, [](auto& elem) {
                elem = (elem > 0 ? 1 : 0);
            });
            return result;
        }

        MatrixX<T> weights;
        VectorX<T> bias, input, output;
        VectorX<T>(*activation)(const VectorX<T>&);
        VectorX<T>(*activation_dx)(const VectorX<T>&);
        const ACTIVATION activationFuncID;
    };


/*
    class PerceptronTrainer {
    public:
        explicit PerceptronTrainer(Perceptron<T>& model_, std::vector<T>&& train, std::vector<T>&& )
        : model(model_) {}

        void train(std::pair<std::span<T>, std::span<T>> trainData, TrainingParams params) {
            std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution dist(0, trainData.size());
            assert(dist.max() == trainData.size() - 1);

            for (size_t epoch = 0; epoch < params.getEpochs(); ++epoch) {
                double epochError = 0;
                for (size_t batch = 0; batch < params.getBatchSize(); ++batch) {
                    const auto& sample = trainData[dist(gen)];
                    epochError += model.updateWeights(sample.first, sample.second, params.getLearningRate());
                }
                std::cout << "Error for epoch " << epoch + 1 << ": " << epochError / params.getBatchSize() << '\n';
            }
        }

    private:
        Perceptron<T>& model;
        std::vector<T> trainInput, trainResult;
    };
*/


    static constexpr auto parseActivationFuncStr = [](std::string_view name) -> ACTIVATION {
        static const std::unordered_map<std::string_view, ACTIVATION> map {
                {"tanh", ACTIVATION::TANH},
                {"sigmoid", ACTIVATION::SIGMOID},
                {"relu", ACTIVATION::RELU},
                {"none", ACTIVATION::NONE}
        };

        if (not map.contains(name))
            throw std::invalid_argument("Unknown activation function: " + std::string{name});

        return map.at(name);
    };

    std::vector<PerceptronLayer> layers;
    size_t inputSize;
};



#endif //PERCEPTRON_PERCEPTRON_H

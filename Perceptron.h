//
// Created by davidl09 on 4/7/24.
//

#ifndef PERCEPTRON_PERCEPTRON_H
#define PERCEPTRON_PERCEPTRON_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <nlohmann/json.hpp>
#include <fstream>

using namespace Eigen;
namespace ranges = std::ranges;
using namespace nlohmann;

template <typename T>
concept Scalar = std::is_floating_point<T>::value;

enum class ACTIVATION {
    SIGMOID,
    RELU,
    TANH,
    NONE,
};


template <Scalar T>
class Perceptron {
public:
    Perceptron(const std::vector<size_t>& shape, bool lastLayerHasActivation = false, std::vector<ACTIVATION> layerFuncs = {}) {
        srand(time(nullptr)); //seed random gen for eigen random();
        //for a network with N layers including input/output, layers_ has length N - 1, and shape has length N.
        if (shape.size() < 3) {
            throw std::invalid_argument(
                    "Network must have at least one hidden layer, has "
                    + std::to_string(shape.size())
                    + " total layers\n"
            );
        }

        layerFuncs.resize(shape.size() - 1, ACTIVATION::TANH);
        if (not lastLayerHasActivation) {
            layerFuncs.back() = ACTIVATION::NONE;
        }

        for (auto i = shape.begin() + 1; i < shape.end(); ++i) {
            layers.push_back(std::move(PerceptronLayer(*i, i[-1], layerFuncs[i - shape.begin() - 1])));
        }
        assert(layers.size() == layerFuncs.size());
    }

    static Perceptron newFromJson(const std::string& filename) {
        json data;
        if (std::ifstream file(filename); file) {
            std::string file_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            try {
                data = json::parse(file_contents);
            }
            catch (std::exception& e) {
                throw std::runtime_error("Could not parse file " + filename + ": " + e.what());
            }
        } else throw std::runtime_error("Could not find file " + filename);

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

        std::vector<ACTIVATION> activations;
        std::vector<size_t> shape;
        shape.push_back(data["inputs"]);

        auto parseActivationFuncStr = [](std::string_view name) -> ACTIVATION {
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

        for (const auto& elem : data["layers"]) {
            shape.push_back(elem["size"].template get<size_t>());
            activations.push_back(parseActivationFuncStr(elem["activation"].template get<std::string>()));
        }

        return Perceptron{shape, true, activations};
    }

    std::vector<T> predict(std::vector<T> input) {
        VectorX<T> result = Map<VectorXd, Unaligned>(input.data(), input.size());

        for (auto& layer : layers) {
            result = layer.propagate(result);
        }

        return {result.begin(), result.end()};
    }

    double updateWeights(const std::vector<T>& data, const std::vector<T>& target, T learningRate = 1e-3) {
        VectorX<T> input = Map<VectorX<T>, Unaligned>(const_cast<T *>(data.data()), data.size()), targetOut = Map<VectorX<T>, Unaligned>(const_cast<T *>(target.data()), target.size());

        for (auto& layer : layers) {
            input = layer.propagate(input);
        }

        auto& output = input;

        VectorX<T> outputError = targetOut - output;
        const double error =  0.5 * pow(outputError.array(), 2).sum();

        for (auto layer = layers.rbegin(); layer < layers.rend(); ++layer) {
            VectorX<T> delta = outputError.cwiseProduct(layer->activation_dx(layer->output)); // Element-wise multiplication
            layer->bias += learningRate * delta; // Update bias

            //MatrixX<T> weightChange = learningRate * delta * layer->input.transpose(); // Weight change
            layer->weights += learningRate * delta * layer->input.transpose(); // Weight change; // Update weights

            outputError = (layer->weights.transpose() * delta).eval(); // Error for the previous layer
        }

        return error;
    }

    friend std::ostream& operator<<(std::ostream& stream, const Perceptron& p) {
        stream << "Layers: " << p.layers.size() << '\n';

        for (const auto& layer : p.layers) {
            stream << layer << '\n';
        }
        return stream;
    }

private:
    struct PerceptronLayer {
    public:
        PerceptronLayer(size_t layerSize, size_t prevLayerSize, ACTIVATION activation = ACTIVATION::SIGMOID)
                : weights(layerSize, prevLayerSize), input(prevLayerSize), output(layerSize), bias(layerSize)
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

        size_t size() const {
            return bias.size();
        }

        friend std::ostream& operator<<(std::ostream& stream, const PerceptronLayer& layer) {
            stream << layer.weights << '\n';
            return stream;
        }

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
    };

    std::vector<PerceptronLayer> layers;

};



#endif //PERCEPTRON_PERCEPTRON_H

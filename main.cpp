#include <iostream>
#include "Perceptron.h"
#include <gnuplot-iostream.h>
#include <valarray>
#include <random>

int main() {
    static constexpr size_t epochs = 5000, batchSize = 100;
    static constexpr double learningRate = 1e-2;

    auto model = Perceptron<double>::newFromJson("model.json");

    std::cout << model;

    std::vector<std::pair<double, double>> values(1000);
    for (int i = 0; i < values.size(); ++i) {
        values[i].first = (static_cast<double>(i) / values.size() - 0.5) * 2 * std::numbers::pi;
        values[i].second = std::pow(values[i].first, 2) / 10;
    }

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, values.size());

    for (int i = 0; i < epochs; ++i) {
        double epochError = 0;
        for (int j = 0; j < batchSize; ++j) {
            auto index = dist(gen);
            epochError += model.updateWeights({values[index].first}, {values[index].second}, learningRate);
        }
        epochError /= batchSize;
        std::cout << "Error for epoch " << i + 1 << ": " << epochError << '\n';
    }

    std::vector<std::pair<double, double>> testData(1000);
    for (int i = 0; i < testData.size(); ++i) {
        testData[i].first = (static_cast<double>(i) / testData.size() - 0.5) * 2 * std::numbers::pi;
        testData[i].second = model.predict({testData[i].first}).front();
    }

    Gnuplot gp;

    gp << "plot '-' with lines title 'Predicted', '-' with lines title 'cos(x)'\n";
    gp.send(testData);
    gp.send(values);

    gp.flush();
/*
    // Keep the plot window open
    std::cout << "Press enter to exit." << std::endl;
    std::cin.get();*/
}

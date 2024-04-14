//
// Created by davidl09 on 4/13/24.
//


#include <iostream>
#include "Perceptron.h"
#include "TrainingParams.h"
#include <gnuplot-iostream.h>
#include <valarray>
#include <random>

int main(int argc, char *argv[]) {

    TrainingParams params{"./training.json"};
    auto model = Perceptron<double>::newFromJson("model.json");

    std::cout << model;

    std::vector<std::pair<double, double>> values(1000);
    for (int i = 0; i < values.size(); ++i) {
        values[i].first = (static_cast<double>(i) / values.size() - 0.5) * 2 * std::numbers::pi;
        values[i].second = std::sin(values[i].first);
    }

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, values.size());

    for (int i = 0; i < params.getEpochs(); ++i) {
        double epochError = 0;
        for (int j = 0; j < params.getBatchSize(); ++j) {
            auto index = dist(gen);
            epochError += model.updateWeights(std::vector{values[index].first}, std::vector{values[index].second}, params.getLearningRate());
        }
        epochError /= params.getBatchSize();
        std::cout << "Error for epoch " << i + 1 << ": " << epochError << '\n';
        if (epochError <= params.getThreshHold()) {
            break;
        }
    }

    std::cout << model;
    model.saveToFolder("mymodel");
    auto newModel = Perceptron<double>::readFromFolder("mymodel");

    std::vector<std::pair<double, double>> testData(1000), loadData(1000);
    for (int i = 0; i < testData.size(); ++i) {
        testData[i].first = (static_cast<double>(i) / testData.size() - 0.5) * 2 * std::numbers::pi;
        testData[i].second = model.predict({testData[i].first}).front();
        loadData[i].first = (static_cast<double>(i) / loadData.size() - 0.5) * 2 * std::numbers::pi;
        loadData[i].second = model.predict({loadData[i].first}).front();
    }

    Gnuplot gp;

    gp << "plot '-' with lines title 'Predicted', '-' with lines title 'After Load', '-' with lines title 'Actual'\n";
    gp.send(testData);
    gp.send(loadData);
    gp.send(values);

    gp.flush();
}

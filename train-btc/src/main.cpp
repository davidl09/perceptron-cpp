//
// Created by davidl09 on 4/14/24.
//

#include <vector>
#include <string_view>
#include <algorithm>

#include "TrainData.h"
#include "Perceptron.h"
#include "TrainingParams.h"
#include "defs.h"

#include <gnuplot-iostream.h>


int main(int argc, char *argv[]) {

    //parse command line args
    std::vector<std::string_view> args{argv, argv + argc};
    args.erase(args.begin());

    //setNbThreads(6);

    auto printHelp = [](){
        std::cout << "Usage: ./train-btc -t <trainingParams.json> -d <data.csv> [-m <modelName.json> -l <existingModelDir>]\n";
        exit(0);
    };

    static constexpr std::string_view
    trainParamsJson = "-t",
    trainDataFile = "-d",
    modelNameSwitch = "-m",
    modelDirSwitch = "-l";

    if (ranges::any_of(args, [](const auto& str) {return str.find("help") != std::string::npos;})) {
        printHelp();
    }

    if (args.empty() or args.size() % 2) {
        std::cerr << "Mismatched command line arguments\n";
        printHelp();
    }

    //we are either constructing a new model or loading an existing one;
    const bool makingNewModel = ranges::any_of(args, [](const auto& str) {
        return str == "-m";
    });
    const bool loadingExistModel = not makingNewModel;

    if (loadingExistModel && not ranges::any_of(args, [](const auto& str) {
        return str == modelDirSwitch;
    })) {
        std::cerr << "Missing either of -l or -m arguments\n";
        printHelp();
    }

    const fs::path modelSource =
            (makingNewModel
            ? std::find(args.begin(), args.end(), modelNameSwitch)[1]
            : std::find(args.begin(), args.end(), modelDirSwitch)[1]
            );

    //check mandatory switches
    for (const auto& s : {trainParamsJson, trainDataFile}) {
        if (not ranges::any_of(args, [s](const auto str) {
            return str == s;
        })) {
            std::cerr << "Missing switch " << s << '\n';
        }
    }

    auto model = (makingNewModel
            ? Perceptron<double>::newFromJson(modelSource)
            : Perceptron<double>::readFromFolder(modelSource));

    TrainData<double> data{ranges::find(args, trainDataFile)[1], model.numInputs(), model.numOutputs()};

    const fs::path trainDataFilePath{std::find(args.begin(), args.end(), trainDataFile)[1]};

    TrainingParams params{ranges::find(args, trainParamsJson)[1]};

    std::vector<std::pair<size_t, double>> errors;
    errors.reserve(params.getEpochs());

    std::vector<decltype(data.getIoPair())> outliers;

    for (int epoch = 0; epoch < params.getEpochs(); ++epoch) {
        double epochError = 0;
        for (int batch = 0; batch < params.getBatchSize(); ++batch) {
            std::pair sample = data.getIoPair(), backup = sample;
            double error = model.updateWeights(std::move(sample.first), std::move(sample.second), params.getLearningRate());
            if (std::round(backup.second[0])) {
                outliers.push_back(std::move(backup));
            }
            epochError += error;
            errors.emplace_back(epoch, error);
        }
        epochError /= static_cast<double>(params.getBatchSize());
        std::cout << "Error for epoch " << epoch + 1 << ": " << epochError << '\n';
    }

    auto ep = params.getEpochs();
    for (auto& outlier : outliers) {
        double error = model.updateWeights(std::move(outlier.first), std::move(outlier.second), params.getLearningRate());
        errors.emplace_back(++ep, error);
        std::cout << "Error for outliers: " << error << '\n';
    }

    model.saveToFolder(loadingExistModel ? modelSource : fs::path(modelSource.string().substr(0, modelSource.string().find('.'))));

    Gnuplot gp;

    gp << "plot '-' with lines title 'Error'\n";
    gp.send(errors);
    gp.flush();

}
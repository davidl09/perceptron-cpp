//
// Created by davidl09 on 4/13/24.
//

#include "Perceptron.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        throw std::invalid_argument("Missing model parameter input file");
    }
    const fs::path jsonPath{argv[1]};

    auto model = Perceptron<double>::newFromJson(jsonPath);
    const auto name = jsonPath.string().substr(0, jsonPath.string().find('.'));
    model.saveToFolder(name);
}

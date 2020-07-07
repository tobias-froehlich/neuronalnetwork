#include <iostream>
#include <random>
#include <ctime>
#include "utils.h"
#include "Neuron.h"
#include "Network.h"
#include "Database.h"
#include "Trainer.h"

Trainer::Trainer(std::string databasefilename) {
  database_.readFromFile(databasefilename);
}

Trainer::~Trainer() {

}

float Trainer::calculate_cost(std::vector<unsigned int> indices, Network & network) {
  float result = 0.0;
  std::vector<float> input{};
  std::vector<float> correct_output{};
  std::vector<float> output{};
  for(unsigned int index : indices) {
    input = database_.getInput(index);
    correct_output = database_.getOutput(index);
    network.reset();
    network.setInput(input);
    output = network.getOutput();
    for(unsigned int i=0; i<output.size(); i++) {
      result += utils::square(output[i] - correct_output[i]);
    }
  }
  return result;
}


void Trainer::StochasticGradientOneBatch(Network & network, unsigned int numberOfSamples, unsigned int numberOfCycles, unsigned int without_last, float factor) {
  std::default_random_engine generator(time(0));
  std::uniform_int_distribution<int> distribution(0, database_.getNumberOfExamples() - 1 - without_last);
  std::vector< std::vector<float> > inputs{};
  std::vector< std::vector<float> > outputs{};
  std::vector< unsigned int > indices;
  for(unsigned int i=0; i<numberOfSamples; i++) {
    int k = distribution(generator);
    indices.push_back(k);
    inputs.push_back(database_.getInput(k));
    outputs.push_back(database_.getOutput(k));
  }
  if (
    (network.getNumberOfInputs() != inputs[0].size())
    || (network.getNumberOfOutputs() != outputs[0].size())
  ) {
    throw std::invalid_argument(
      "Number of in and outputs of network does not fit to database."
    );
  }


  utils::log(std::string("factor=") + std::to_string(factor));
  utils::log(std::string("numberOfSamples=") + std::to_string(numberOfSamples));
  for(unsigned int cycle=0; cycle<numberOfCycles; cycle++) {
    float cost_old = calculate_cost(indices, network);
    network.resetDelta();
    for(unsigned int i=0; i<numberOfSamples; i++) {
      network.reset();
      network.BackPropagation(inputs[i], outputs[i]);
      network.addGradientsToDelta();
    }
    network.applyDelta(-factor);
    float cost = calculate_cost(indices, network);
    utils::log(std::string("cycle=") + std::to_string(cycle) + std::string(" cost=") + std::to_string(cost) + std::string(" factor=") + std::to_string(factor));
  }
  

}

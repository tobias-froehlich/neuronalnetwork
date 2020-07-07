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
/*
unsigned int Trainer::number_of_correct(std::vector<unsigned int> indices, Network & network) {
  unsigned int result = 0;
  std::vector<float> input{};
  std::vector<float> correct_output{};
  std::vector<float> output{};
  for(unsigned int index : indices) {
    input = database_.getInput(index);
    correct_output = database_.getOutput(index);
    network.reset();
    network.setInput(input);
    output = network.getOutput();
    if (utils::index_at_max(output) == utils::index_at_max(correct_output)) {
      result++;
    }
  }
  return result;
}
*/
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
/*
void Trainer::trainOutputOnly(Network & network, unsigned int number_of_cycles, unsigned int number_of_samples, float randomizefactor) {
  Network network_copy(network);
  
  std::default_random_engine generator(time(0));
  std::uniform_int_distribution<int> distribution(0, database_.getNumberOfExamples() - 1);

  std::vector<unsigned int> indices{};
  for(unsigned int i=0; i<number_of_samples; i++) {
    indices.push_back(distribution(generator));
  }

  float cost = calculate_cost(indices, network);
  float cost_old = cost;
  bool better = false;
  for(unsigned int cycle=0; cycle < number_of_cycles; cycle++) {
    indices.clear();
    std::vector<unsigned int> indices{};
    for(unsigned int i=0; i<number_of_samples; i++) {
      indices.push_back(distribution(generator));
    }
    better = false;
    while ( ! better ) {
      std::cout << "cycle: " << cycle << ", cost: " << cost_old << '\n';
      cost_old = calculate_cost(indices, network);
      network.passParameters(network_copy);
      network_copy.randomizeOutputBiases(randomizefactor);
      network_copy.randomizeOutputWeights(randomizefactor);
      cost = calculate_cost(indices, network_copy);
      if (cost <= cost_old) {
        better = true;
        cost_old = cost;
        network_copy.passParameters(network);
        std::cout << "                          correct: " << number_of_correct(indices, network) << '\n';
      }
    }
  }
}
*/

/*
void Trainer::train(Network & network, unsigned int number_of_cycles, unsigned int number_of_samples, float randomizefactor) {
  Network network_copy(network);
  
  std::default_random_engine generator;
  generator.seed(time(0));
  std::uniform_int_distribution<unsigned int> distribution(0, database_.getNumberOfExamples() - 1);

  unsigned int N = 0;
  unsigned int correct1 = 0;
  unsigned int correct2 = 0;
  std::vector<unsigned int> indices{distribution(generator)};
  network_copy.randomizeWeights(randomizefactor);
  network_copy.randomizeBiases(randomizefactor);
  for(unsigned int cycle=0; cycle<number_of_cycles; cycle++) {
    network.passParameters(network_copy);
    network_copy.randomizeWeights(randomizefactor);
    network_copy.randomizeBiases(randomizefactor);
    correct1 = 0;
    correct2 = 0;
    correct1 += number_of_correct(indices, network);
    correct2 += number_of_correct(indices, network_copy);
    N = 1;
    while (abs(correct2) < 3*sqrt(N)) {
      indices[0] = distribution(generator);
      correct1 += number_of_correct(indices, network);
      correct2 += number_of_correct(indices, network_copy);
      N += 1;
//      std::cout << N << ' ' << correct1 << ' ' << correct2 << '\n';
    }
    if (correct2 > correct1) {
      network_copy.passParameters(network);
      std::cout << "                         * \n";
    }
    std::cout << N << "         " << (float)correct1 / float(N) << '\n';
  }
*/

//  Network network_copy(network);
//  
//  std::default_random_engine generator;
//  std::uniform_int_distribution<int> distribution(0, mnist_.getNumberOfExamples() - 1);
//
//  std::vector<unsigned int> indices{};
//  for(unsigned int i=0; i<number_of_samples; i++) {
//    indices.push_back(distribution(generator));
//  }
//
//  float cost = calculate_cost(indices, network);
//  float cost_old = cost;
//  bool better = false;
//  for(unsigned int cycle=0; cycle < number_of_cycles; cycle++) {
//    indices.clear();
//    std::vector<unsigned int> indices{};
//    for(unsigned int i=0; i<number_of_samples; i++) {
//      indices.push_back(distribution(generator));
//    }
//    better = false;
//    while ( ! better ) {
//      std::cout << "cycle: " << cycle << ", cost: " << cost_old << '\n';
//      cost_old = calculate_cost(indices, network);
//      network.passParameters(network_copy);
//      network_copy.randomizeBiases(randomizefactor);
//      network_copy.randomizeWeights(randomizefactor);
//      cost = calculate_cost(indices, network_copy);
//      if (cost <= cost_old) {
//        better = true;
//        cost_old = cost;
//        network_copy.passParameters(network);
//        std::cout << "                          correct: " << number_of_correct(indices, network) << '\n';
//      }
//    }
//  }
//}

/*
void Trainer::reduce(Network & network, unsigned int number_of_samples) {
  unsigned int best_for_reduction = network.getNumberOfNeurons();
  float best_cost = 1000000.0;

  std::default_random_engine generator;
  generator.seed(time(0));
  std::uniform_int_distribution<int> distribution(0, database_.getNumberOfExamples() - 1);

  std::vector<unsigned int> indices{};
  for(unsigned int i=0; i<number_of_samples; i++) {
    indices.push_back(distribution(generator));
  }

  for(unsigned int i=0; i<network.getNumberOfNeurons(); i++) {
    if ( ! (network.isInput(network.getNeuronByIndex(i)) || network.isOutput(network.getNeuronByIndex(i)))){
      float cost = 0.0;
      std::vector<float> input{};
      std::vector<float> correct_output{};
      std::vector<float> output{};
      for(unsigned int index : indices) {
        input = database_.getInput(index);
        correct_output = database_.getOutput(index);
        network.reset();
        network.getNeuronByIndex(i)->forceOutput(0.0);
        network.setInput(input);
        output = network.getOutput();
        for(unsigned int i=0; i<output.size(); i++) {
          cost += utils::square(output[i] - correct_output[i]);
        }
      }
      if (cost < best_cost) {
        best_cost = cost;
        best_for_reduction = i;
      }
    }
  }
  network.removeNeuronByIndex(best_for_reduction);
  std::cout << "new cost: " << best_cost << '\n';
}
*/


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
/*    if (cycle % 2 == 0) {
      network.applyDeltaBiasesOnly(-factor);
    }
    else {
      network.applyDeltaWeightsOnly(-factor);
    }
*/
    float cost = calculate_cost(indices, network);
    utils::log(std::string("cycle=") + std::to_string(cycle) + std::string(" cost=") + std::to_string(cost) + std::string(" factor=") + std::to_string(factor));
  }
  

}

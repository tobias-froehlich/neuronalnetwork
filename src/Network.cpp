#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include "utils.h"
#include "Neuron.h"
#include "Network.h"

Network::Network() {
  generator_.seed(time(0));
}

Network::Network(const Network & other) : Network() {
  unsigned int n = other.getNumberOfNeurons();
  for(unsigned int i=0; i < n; i++) {
    createNeuron();
  }
  for(unsigned int i=0; i<other.getNumberOfInputs(); i++) {
    addInput(getNeuronByIndex(other.getIndexByNeuron(other.getInputByIndex(i))));
  }
  for(unsigned int i=0; i<other.getNumberOfOutputs(); i++) {
    addOutput(getNeuronByIndex(other.getIndexByNeuron(other.getOutputByIndex(i))));
  }
  for(unsigned int i=0; i < n; i++) {
    Neuron* neuron = other.getNeuronByIndex(i);
    getNeuronByIndex(i)->setBias(neuron->getBias());
    unsigned int numberOfSources = neuron->getNumberOfSources();
    for(unsigned int j=0; j < numberOfSources; j++) {
      connect(other.getIndexByNeuron(neuron->getSource(j)), i, neuron->getWeight(j));
    }
  }
}

Network::Network(std::vector<unsigned int> layers) : Network() {
  for(unsigned int i=0; i<layers[0]; i++) {
    createNeuron();
    addInput(getNeuronByIndex(i));
  }
  unsigned int first_index_old = 0;
  unsigned int first_index = 0;
  for(unsigned int l=1; l<layers.size(); l++) {
    first_index += layers[l-1];
    for(unsigned int i=0; i<layers[l]; i++) {
      createNeuron();
      for(unsigned int j=0; j<layers[l-1]; j++) {
        connect(first_index_old + j, first_index + i, 0.0);
      }
    }
    first_index_old = first_index;
  }
  for(unsigned int i=getNumberOfNeurons() - layers.back(); i<getNumberOfNeurons(); i++) {
    addOutput(getNeuronByIndex(i));
  }
}

Network::~Network() {
  for(unsigned int i=0; i<numberOfNeurons_; i++) {
    delete neurons_[i];
  }
}

void Network::passParameters(Network& other) const {
  unsigned int n = 0;
  for(unsigned int i=0; i<numberOfNeurons_; i++) {
    Neuron neuron = *neurons_[i];
    other.getNeuronByIndex(i)->setBias(neuron.getBias());
    n = neuron.getNumberOfSources();
    for(unsigned int j=0; j<n; j++) {
      other.getNeuronByIndex(i)->setWeight(j, neuron.getWeight(j));
    }
  }
}

unsigned int Network::getNumberOfNeurons() const {
  return numberOfNeurons_;
}

unsigned int Network::getNumberOfInputs() const{
  return numberOfInputs_;
}

unsigned int Network::getNumberOfOutputs() const {
  return numberOfOutputs_;
}

Neuron* Network::getNeuronByIndex(unsigned int index) const {
  if (index >= numberOfNeurons_) {
    throw std::invalid_argument(
      "Index exceeds number of neurons."
    );
  }
  return neurons_[index];
}

unsigned int Network::getIndexByNeuron(Neuron* neuron) const {
  for(unsigned int i=0; i < numberOfNeurons_; i++) {
    if (neurons_[i] == neuron) {
      return i;
    }
  }
  throw std::invalid_argument(
    "Neuron does not exist in the network."
  );
}

bool Network::isInput(Neuron* neuron) const {
  for(Neuron* n : inputs_) {
    if (n == neuron) {
      return true;
    }
  }
  return false;
}

bool Network::isOutput(Neuron* neuron) const {
  for(Neuron* n : outputs_) {
    if (n == neuron) {
      return true;
    }
  }
  return false;
}

void Network::createNeuron() {
  Neuron* neuron = new Neuron();
  neurons_.push_back(neuron);
  numberOfNeurons_++;
}

void Network::addInput(Neuron* neuron) {
  inputs_.push_back(neuron);
  numberOfInputs_++;
}

Neuron* Network::getInputByIndex(unsigned int index) const {
  if (index >= numberOfInputs_) {
    throw std::invalid_argument(
      "Index exeeds number of inputs."
    );
  }
  return inputs_[index];
}

void Network::addOutput(Neuron* neuron) {
  outputs_.push_back(neuron);
  numberOfOutputs_++;
}

Neuron* Network::getOutputByIndex(unsigned int index) const {
  if (index >= numberOfOutputs_) {
    throw std::invalid_argument(
      "Index exeeds number of outputs."
    );
  }
  return outputs_[index];
}

void Network::removeNeuronByIndex(unsigned int index) {
  if (index >= numberOfNeurons_) {
    throw std::invalid_argument(
      "Index exceeds number of neurons."
    );
  }
  Neuron* neuron = neurons_[index];
  neurons_.erase(neurons_.begin() + index);
  numberOfNeurons_--;
  for(unsigned int i=0; i<numberOfNeurons_; i++) {
    neurons_[i]->removeSource(neuron);
  }
  for(unsigned int i=0; i<numberOfInputs_; i++) {
    if (inputs_[i] == neuron) {
      inputs_.erase(inputs_.begin() + i);
      i--;
      numberOfInputs_--;
    }
  }
  for(unsigned int i=0; i<numberOfOutputs_; i++) {
    if (outputs_[i] == neuron) {
      outputs_.erase(outputs_.begin() + i);
      i--;
      numberOfOutputs_--;
    }
  }
  delete neuron;
}

void Network::connect(
  unsigned int index_from,
  unsigned int index_to,
  float weight
) {
  if (index_from >= numberOfNeurons_) {
    throw std::invalid_argument(
      "Index exceeds number of neurons."
    );
  }
  if (index_to >= numberOfNeurons_) {
    throw std::invalid_argument(
      "Index exceeds number of neurons."
    );
  }
  neurons_[index_to]->addSource(
    neurons_[index_from],
    weight
  );
}

void Network::reset() {
  for(unsigned int i=0; i < numberOfNeurons_; i++) {
    neurons_[i]->reset();
  }
}

void Network::setInput(std::vector<float> input) {
  if (input.size() != numberOfInputs_) {
    throw std::invalid_argument(
      "Number of inputs in setInput() not correct."
    );
  }

  for(unsigned int i=0; i < numberOfInputs_; i++) {
    inputs_[i]->forceOutput(input[i]);
  }
}

std::vector<float> Network::getOutput() {
  std::vector<float> output{};
  for(unsigned int i=0; i < numberOfOutputs_; i++) {
    output.push_back(outputs_[i]->getOutput());
  }
  return output;
}

void Network::readFromFile(std::string filename) {
  std::ifstream file(filename);
  if ( ! file ) {
    throw std::invalid_argument(
      "Cannot open file for reading."
    );
  }

  std::string line;
  unsigned int line_index = 0;
  std::vector<std::string> words;
  unsigned int index = 0;
  while (getline(file, line)) {
    line_index += 1;
    words = utils::split(line, ' ');
    if (line_index == 1) {
      unsigned int number_of_neurons = std::stoi(words[0]);
      for(unsigned int i=0; i < number_of_neurons; i++) {
        createNeuron();
      }
    }
    else if (line_index == 2) {
      for(unsigned int i=0; i < words.size(); i++) {
        addInput(getNeuronByIndex(std::stoi(words[i])));
      }
    }
    else if (line_index == 3) {
      for(unsigned int i=0; i < words.size(); i++) {
        addOutput(getNeuronByIndex(std::stoi(words[i])));
      }
    }
    else {
      index = std::stoi(words[0]);
      getNeuronByIndex(index)->setBias(std::stof(words[1]));
      for(unsigned int i=2; i<words.size(); i++) {
        if (i % 2 == 0) {
          getNeuronByIndex(index)->addSource(getNeuronByIndex(std::stoi(words[i])), std::stof(words[i+1]));
        }
      }
    }
  }

  file.close();
}

void Network::writeToFile(std::string filename) const {
  std::ofstream file(filename);
  if ( ! file ) {
    throw std::invalid_argument(
      "Cannot open file for writing."
    );
  }
  file << numberOfNeurons_ << '\n';
  for(unsigned int i=0; i < numberOfInputs_; i++) {
    file << getIndexByNeuron(inputs_[i]) << " ";
  }
  file << '\n';
  for(unsigned int i=0; i < numberOfOutputs_; i++) {
    file << getIndexByNeuron(outputs_[i]) << " ";
  }
  file << '\n';
  for(unsigned int i=0; i < numberOfNeurons_; i++) {
    file << i << ' ';
    file << neurons_[i]->getBias() << ' ';
    for(unsigned j=0; j < neurons_[i]->getNumberOfSources(); j++) {
      file << getIndexByNeuron(neurons_[i]->getSource(j)) << ' ';
      file << neurons_[i]->getWeight(j) << ' ';
    }
    file << '\n';
  }
  file.close();
}

void Network::randomizeWeights(float factor) {
  Neuron* neuron = NULL;
  unsigned int n = 0;
  factor *= distribution_(generator_);
  for(unsigned int i=0; i < numberOfNeurons_; i++) {
    neuron = neurons_[i];
    n = neuron->getNumberOfSources();
    for(unsigned int j=0; j < n; j++) {
      neuron->setWeight(j, neuron->getWeight(j) + distribution_(generator_)*factor);
    }
  }
}

void Network::randomizeBiases(float factor) {
  factor *= distribution_(generator_);
  for(unsigned int i=0; i < numberOfNeurons_; i++) {
    neurons_[i]->setBias(neurons_[i]->getBias() + distribution_(generator_)*factor);
  }
}

void Network::randomizeOutputWeights(float factor) {
  Neuron* neuron = NULL;
  unsigned int n = 0;
  for(unsigned int i=0; i < numberOfOutputs_; i++) {
    neuron = outputs_[i];
    n = neuron->getNumberOfSources();
    for(unsigned int j=0; j < n; j++) {
      neuron->setWeight(j, neuron->getWeight(j) + distribution_(generator_)*factor);
    }
  }
}

void Network::randomizeOutputBiases(float factor) {
  for(unsigned int i=0; i < numberOfOutputs_; i++) {
    outputs_[i]->setBias(neurons_[i]->getBias() + distribution_(generator_)*factor);
  }
}

void Network::resetDelta() {
  for(Neuron* & neuron : neurons_) {
    neuron->resetDelta();
  }
}

void Network::resetGradients() {
  for(Neuron* & neuron : neurons_) {
    neuron->resetGradients();
  }
}

void Network::addGradientsToDelta() {
  for(Neuron* & neuron : neurons_) {
    neuron->addGradientToDelta();
  }
}

float Network::absDelta() {
  float result = 0.0;
  for(Neuron* & neuron : neurons_) {
    result += neuron->sumOfDeltaSquared();
  }
  return sqrt(result);
}

void Network::applyDelta(float factor) {
  for(Neuron* & neuron : neurons_) {
    neuron->applyDelta(factor);
  }
}

void Network::applyDeltaBiasesOnly(float factor) {
  for(Neuron* & neuron : neurons_) {
    neuron->applyDeltaBiasOnly(factor);
  }
}

void Network::applyDeltaWeightsOnly(float factor) {
  for(Neuron* & neuron : neurons_) {
    neuron->applyDeltaWeightsOnly(factor);
  }
}


void Network::BackPropagation(std::vector<float> input, std::vector<float> output) {
  reset();
  resetGradients();
  setInput(input);
  for(unsigned int i=0; i<numberOfOutputs_; i++) {
    outputs_[i]->calculateOutputGradientOutput(output[i]);
  }
  for(unsigned int i=0; i<numberOfNeurons_; i++) {
    for(unsigned int j=0; j<neurons_[i]->getNumberOfSources(); j++) {
      neurons_[i]->getWeightGradient(j);
    }
    neurons_[i]->getBiasGradient();
  }
}

void Network::writePythonScript(std::string filename) {
  std::ofstream file(filename);
  if (! file) {
    throw std::invalid_argument(
      "Could not open file for writing python script."
    );
  }
  file << "from activation_function import activation_function as f\n";
  file << "def network(input):\n";
  file << "    a = [0.0] * " << numberOfNeurons_ << "\n";
  unsigned int count = 0;
  for(Neuron* neuron : neurons_) {
    neuron->reset();
    neuron->setPythonVarName(std::string("a[") + std::to_string(count) + std::string("]"));
    count++;
  }
  count = 0;
  for(Neuron* neuron : inputs_) {
    file << "    " << neuron->getPythonVarName() << " = " << "input[" << count << "]\n";
    count++;
  }
  for(Neuron* neuron : outputs_) {
    neuron->getPythonScript(file);
  }
  file << "    output = [";
  count = 0;
  for(Neuron* neuron : outputs_) {
    file << neuron->getPythonVarName() << ", ";
    count++;
  }
  file << "]\n";
  file << "    return output\n";
  file.close();
}

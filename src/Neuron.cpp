#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "utils.h"
#include "Neuron.h"

Neuron::Neuron() {
//  std::ofstream file;
//  file.open("/tmp/log.txt", std::ios_base::app);
//  file << "*";
//  file.close();
}

Neuron::~Neuron() {
//  std::ofstream file;
//  file.open("/tmp/log.txt", std::ios_base::app);
//  file << "+";
//  file.close();

}

unsigned int Neuron::getNumberOfSources() {
  return number_of_sources_;
}

unsigned int Neuron::getNumberOfDrains() {
  return number_of_drains_;
}

Neuron* Neuron::getSource(unsigned int index) {
  if (index >= number_of_sources_) {
    throw std::invalid_argument(
      "Index exeeds number of sources."
    );
  }
  return source_neurons_[index];
}

Neuron* Neuron::getDrain(unsigned int index) {
  return drain_neurons_[index];
}

unsigned int Neuron::getDrainSourceIndex(unsigned int index) {
  return drainSourceIndices_[index];
}

void Neuron::setWeight(unsigned int index, float weight) {
  if (index >= number_of_sources_) {
    throw std::invalid_argument(
      "Index exeeds number of sources."
    );
  }
  weights_[index] = weight;
}


float Neuron::getWeight(unsigned int index) {
  if (index >= number_of_sources_) {
    throw std::invalid_argument(
      "Index exeeds number of sources."
    );
  }
  return weights_[index];
}

void Neuron::addSource(Neuron* neuron, float weight) {
  source_neurons_.push_back(neuron);
  weights_.push_back(weight);
  weightsGradient_.push_back(0.0);
  weightsDelta_.push_back(0.0);
  neuron->addDrain(this, number_of_sources_);
  number_of_sources_++;
}

void Neuron::addDrain(Neuron* neuron, float drainSourceIndex) {
  drain_neurons_.push_back(neuron);
  drainSourceIndices_.push_back(drainSourceIndex);
  number_of_drains_++;
}

void Neuron::removeSourceByIndex(unsigned int index) {
  if (index >= number_of_sources_) {
    throw std::invalid_argument(
      "Cannot remove source at that index."
    );
  }
  Neuron* neuron = source_neurons_[index];
  neuron->removeDrain(this);
  source_neurons_.erase(source_neurons_.begin() + index);
  weights_.erase(weights_.begin() + index);
  weightsGradient_.erase(weightsGradient_.begin() + index);
  weightsDelta_.erase(weightsDelta_.begin() + index);
  number_of_sources_--;
}

void Neuron::removeSource(Neuron* neuron) {
  for(unsigned int i=0; i<number_of_sources_; i++) {
    if (source_neurons_[i] == neuron) {
      removeSourceByIndex(i);
    }
  }
}

void Neuron::removeDrainByIndex(unsigned int index) {
  drain_neurons_.erase(drain_neurons_.begin() + index);
  drainSourceIndices_.erase(drainSourceIndices_.begin() + index);
  number_of_drains_--;
}


void Neuron::removeDrain(Neuron* neuron) {
  for(unsigned int i=0; i<number_of_drains_; i++) {
    if (drain_neurons_[i] == neuron) {
      removeDrainByIndex(i);
    }
  }
}

void Neuron::forceOutput(float output) {
  output_ = output;
  set_ = true;
}

float Neuron::getOutput() {
  if (set_) {
    return output_;
  }
  else {
    output_ = 0.0;
    for(unsigned int i=0; i<number_of_sources_; i++) {
      output_ +=
        source_neurons_[i]->getOutput()
        * weights_[i];
    }
    outputWithoutActivationFunction_ = output_ + bias_;
    output_ = utils::activation_function(outputWithoutActivationFunction_);
    set_ = true;
    return output_;
  }
}

float Neuron::getOutputWithoutActivationFunction() {
  if ( ! set_ ) {
    getOutput();
  }
  return outputWithoutActivationFunction_;
};

void Neuron::reset() {
  set_ = false;
}

void Neuron::setBias(float bias) {
  bias_ = bias;
}

float Neuron::getBias() {
  return bias_;
}

void Neuron::resetGradients() {
//  std::cout << "nueuron reset gradient\n";
  for(unsigned int i=0; i<number_of_sources_; i++) {
    weightsGradient_[i] = 0.0;
  }
  biasGradient_ = 0.0;
  outputGradient_ = 0.0;
  weightGradientSet_ = false;
  biasGradientSet_ = false;
  outputGradientSet_ = false;
}

void Neuron::resetDelta() {
  for(unsigned int i=0; i<number_of_sources_; i++) {
    weightsDelta_[i] = 0.0;
  }
  biasDelta_ = 0.0;
}

void Neuron::calculateOutputGradientOutput(float correct_output) {
  outputGradient_ = 2.0 * (getOutput() - correct_output);
  outputGradientSet_ = true;
}

float Neuron::getOutputGradient() {
  if ( ! outputGradientSet_ ) {
    outputGradient_ = 0.0;
    for(unsigned int i=0; i<number_of_drains_; i++) {
      Neuron* drain = getDrain(i);
      outputGradient_ += utils::activation_function_derivative(drain->getOutputWithoutActivationFunction()) * drain->getWeight(getDrainSourceIndex(i)) * drain->getOutputGradient();
    }
    outputGradientSet_ = true;
  }
  return outputGradient_;
}

float Neuron::getWeightGradient(unsigned int index) {
  if ( ! weightGradientSet_ ) {
    for(unsigned int i=0; i < number_of_sources_; i++) {
      Neuron* source = getSource(i);
      weightsGradient_[i] = utils::activation_function_derivative(getOutputWithoutActivationFunction()) * source->getOutput() * getOutputGradient();
    }
    weightGradientSet_ = true;
  }
  return weightsGradient_[index];
};

float Neuron::getBiasGradient() {
  if ( ! biasGradientSet_ ) {
    biasGradient_ = utils::activation_function_derivative(getOutputWithoutActivationFunction()) * getOutputGradient();
    biasGradientSet_ = true;
  }
  return biasGradient_;
}

void Neuron::addGradientToDelta() {
  for(unsigned int i=0; i<number_of_sources_; i++) {
    weightsDelta_[i] += weightsGradient_[i];
  }
  biasDelta_ += biasGradient_;
}

float Neuron::sumOfDeltaSquared() {
  float result = utils::square(biasDelta_);
  for(float weightdelta : weightsDelta_) {
    result += utils::square(weightdelta);
  }
  return result;
}

void Neuron::applyDelta(float factor) {
  for(unsigned int i=0; i<number_of_sources_; i++) {
    weights_[i] += weightsDelta_[i] * factor;
    if (weights_[i] > 1.0) {weights_[i] = 1.0;}
    if (weights_[i] < -1.0) {weights_[i] = -1.0;}
  }
  
  bias_ += biasDelta_ * factor;
  if (bias_ > 1.0) {bias_ = 1.0;}
  if (bias_ < -1.0) {bias_ = -1.0;}
}

void Neuron::reduceWeights() {
  if (number_of_sources_ > 32) {
    unsigned int i_weakest = 0;
    float weight_weakest = 1000000;
    
    for(unsigned int i=0; i<number_of_sources_; i++) {
      if (abs(weights_[i]) < weight_weakest) {
        weight_weakest = abs(weights_[i]);
        i_weakest = i;
      }
    }
    removeSourceByIndex(i_weakest);
  }
}

void Neuron::print() {
  std::cout << "    Neuron at " << this << '\n';
  std::cout << "        number_of_sources_ = " << number_of_sources_ << '\n';
  std::cout << "        number_of_drains_ = " << number_of_drains_ << '\n';
  std::cout << "        source_neurons_ = [";
  for(Neuron* neuron: source_neurons_) {
    std::cout << neuron << ", ";
  }
  std::cout << "]\n";
  std::cout << "        weights_ = [";
  for (float w : weights_) {
    std::cout << w << ", ";
  }
  std::cout << "]\n";
}

void Neuron::setPythonVarName(std::string name) {
  pythonVarName_ = name;
}

std::string Neuron::getPythonVarName() {
  set_ = true;
  return pythonVarName_;
}

void Neuron::getPythonScript(std::ostream & os) {
  if ( ! set_ ) {
    for(Neuron* neuron : source_neurons_) {
      neuron->getPythonScript(os);
    }
    os << "    " << pythonVarName_ << " = f(";
    for(unsigned int i=0; i<number_of_sources_; i++) {
      Neuron* neuron = source_neurons_[i];
      os << weights_[i] << " * " << neuron->getPythonVarName() << " + ";
    }
    os << bias_;
    os << ")\n";
  }
  set_ = true;
}

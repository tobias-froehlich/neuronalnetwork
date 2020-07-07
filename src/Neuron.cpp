#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "utils.h"
#include "Neuron.h"

Neuron::Neuron() {

}

Neuron::~Neuron() {

}

unsigned int Neuron::getNumberOfSources() {
  return numberOfSources_;
}

unsigned int Neuron::getNumberOfDrains() {
  return numberOfDrains_;
}

Neuron* Neuron::getSource(unsigned int index) {
  if (index >= numberOfSources_) {
    throw std::invalid_argument(
      "Index exeeds number of sources."
    );
  }
  return sourceNeurons_[index];
}

Neuron* Neuron::getDrain(unsigned int index) {
  return drainNeurons_[index];
}

unsigned int Neuron::getDrainSourceIndex(unsigned int index) {
  return drainSourceIndices_[index];
}

void Neuron::setWeight(unsigned int index, float weight) {
  if (index >= numberOfSources_) {
    throw std::invalid_argument(
      "Index exeeds number of sources."
    );
  }
  weights_[index] = weight;
}


float Neuron::getWeight(unsigned int index) {
  if (index >= numberOfSources_) {
    throw std::invalid_argument(
      "Index exeeds number of sources."
    );
  }
  return weights_[index];
}

void Neuron::addSource(Neuron* neuron, float weight) {
  sourceNeurons_.push_back(neuron);
  weights_.push_back(weight);
  weightsGradient_.push_back(0.0);
  weightsDelta_.push_back(0.0);
  neuron->addDrain(this, numberOfSources_);
  numberOfSources_++;
}

void Neuron::addDrain(Neuron* neuron, float drainSourceIndex) {
  drainNeurons_.push_back(neuron);
  drainSourceIndices_.push_back(drainSourceIndex);
  numberOfDrains_++;
}

void Neuron::removeSourceByIndex(unsigned int index) {
  if (index >= numberOfSources_) {
    throw std::invalid_argument(
      "Cannot remove source at that index."
    );
  }
  Neuron* neuron = sourceNeurons_[index];
  neuron->removeDrain(this);
  sourceNeurons_.erase(sourceNeurons_.begin() + index);
  weights_.erase(weights_.begin() + index);
  weightsGradient_.erase(weightsGradient_.begin() + index);
  weightsDelta_.erase(weightsDelta_.begin() + index);
  numberOfSources_--;
}

void Neuron::removeSource(Neuron* neuron) {
  for(unsigned int i=0; i<numberOfSources_; i++) {
    if (sourceNeurons_[i] == neuron) {
      removeSourceByIndex(i);
    }
  }
}

void Neuron::removeDrainByIndex(unsigned int index) {
  drainNeurons_.erase(drainNeurons_.begin() + index);
  drainSourceIndices_.erase(drainSourceIndices_.begin() + index);
  numberOfDrains_--;
}


void Neuron::removeDrain(Neuron* neuron) {
  for(unsigned int i=0; i<numberOfDrains_; i++) {
    if (drainNeurons_[i] == neuron) {
      removeDrainByIndex(i);
    }
  }
}

void Neuron::forceOutput(float output) {
  output_ = output;
  OutputSet_ = true;
}

float Neuron::getOutput() {
  if (OutputSet_) {
    return output_;
  }
  else {
    output_ = 0.0;
    for(unsigned int i=0; i<numberOfSources_; i++) {
      output_ +=
        sourceNeurons_[i]->getOutput()
        * weights_[i];
    }
    outputWithoutActivationFunction_ = output_ + bias_;
    output_ = utils::activation_function(outputWithoutActivationFunction_);
    OutputSet_ = true;
    return output_;
  }
}

float Neuron::getOutputWithoutActivationFunction() {
  if ( ! OutputSet_ ) {
    getOutput();
  }
  return outputWithoutActivationFunction_;
};

void Neuron::resetOutput() {
  OutputSet_ = false;
}

void Neuron::setBias(float bias) {
  bias_ = bias;
}

float Neuron::getBias() {
  return bias_;
}

void Neuron::resetGradients() {
  for(unsigned int i=0; i<numberOfSources_; i++) {
    weightsGradient_[i] = 0.0;
  }
  biasGradient_ = 0.0;
  outputGradient_ = 0.0;
  weightGradientSet_ = false;
  biasGradientSet_ = false;
  outputGradientSet_ = false;
}

void Neuron::resetDelta() {
  for(unsigned int i=0; i<numberOfSources_; i++) {
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
    for(unsigned int i=0; i<numberOfDrains_; i++) {
      Neuron* drain = getDrain(i);
      outputGradient_ += utils::activation_function_derivative(drain->getOutputWithoutActivationFunction()) * drain->getWeight(getDrainSourceIndex(i)) * drain->getOutputGradient();
    }
    outputGradientSet_ = true;
  }
  return outputGradient_;
}

float Neuron::getWeightGradient(unsigned int index) {
  if ( ! weightGradientSet_ ) {
    for(unsigned int i=0; i < numberOfSources_; i++) {
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
  for(unsigned int i=0; i<numberOfSources_; i++) {
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
  applyDeltaBiasOnly(factor);
  applyDeltaWeightsOnly(factor);
}

void Neuron::applyDeltaBiasOnly(float factor) {
  bias_ += biasDelta_ * factor;
}

void Neuron::applyDeltaWeightsOnly(float factor) {
  for(unsigned int i=0; i<numberOfSources_; i++) {
    weights_[i] += weightsDelta_[i] * factor;
  }
}

void Neuron::reduceWeights() {
  if (numberOfSources_ > 32) {
    unsigned int i_weakest = 0;
    float weight_weakest = 1000000;
    
    for(unsigned int i=0; i<numberOfSources_; i++) {
      if (abs(weights_[i]) < weight_weakest) {
        weight_weakest = abs(weights_[i]);
        i_weakest = i;
      }
    }
    removeSourceByIndex(i_weakest);
  }
}

void Neuron::setPythonVarName(std::string name) {
  pythonVarName_ = name;
}

std::string Neuron::getPythonVarName() {
  OutputSet_ = true;
  return pythonVarName_;
}

void Neuron::getPythonScript(std::ostream & os) {
  if ( ! OutputSet_ ) {
    for(Neuron* neuron : sourceNeurons_) {
      neuron->getPythonScript(os);
    }
    os << "    " << pythonVarName_ << " = f(";
    for(unsigned int i=0; i<numberOfSources_; i++) {
      Neuron* neuron = sourceNeurons_[i];
      os << weights_[i] << " * " << neuron->getPythonVarName() << " + ";
    }
    os << bias_;
    os << ")\n";
  }
  OutputSet_ = true;
}

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "utils.h"
#include "Database.h"

void Database::readFromFile(std::string filename) {
  inputs_.clear();
  outputs_.clear();
  numberOfExamples_ = 0;
  std::ifstream file(filename);
  if (! file) {
    throw std::invalid_argument(
      "Database file not found."
    );
  }
  unsigned int number_of_inputs = 0;
  unsigned int number_of_outputs = 0;
  unsigned int number_of_examples = 0;
  file >> number_of_inputs;
  file >> number_of_outputs;
  file >> number_of_examples;
  std::cout << "number of inputs : " << number_of_inputs << '\n';
  std::cout << "number of outputs : " << number_of_outputs << '\n';
  std::cout << "number of examples : " << number_of_examples << '\n';
  float number = 0.0;
  for(unsigned int e=0; e<number_of_examples; e++) {
    inputs_.push_back({});
    outputs_.push_back({});
    for(unsigned int i=0; i<number_of_inputs; i++) {
      file >> number;
      inputs_.back().push_back(number);
    }
    for(unsigned int a=0; a<number_of_outputs; a++) {
      file >> number;
      outputs_.back().push_back(number);
    }
    numberOfExamples_++;
  }
  file.close();

/*  std::ifstream outputfile("../preparation/output.txt");
  std::vector< std::vector<float> > outputimages{};
  std::string line;
  std::vector<std::string> words;
  while(getline(outputfile, line)) {
    words = utils::split(line, ' ');
    std::vector<float> outputimage{};
    for(std::string word : words) {
      outputimage.push_back(std::stof(word));
    }
    outputimages.push_back(outputimage);
  }
  outputfile.close();
  std::ifstream file(filename);
  if ( ! file ) {
    throw std::invalid_argument(
      "File does not exist for reading MNIST database."
    );
  }
  float one_over_255 = 1.0 / 255.0;
  int number;
  while (getline(file, line)) {
    words = utils::split(line, ',');
    number = std::stoi(words[0]);
    numbers_.push_back(number);
    outputs_.push_back(outputimages[number]);
    inputs_.push_back({});
    for(unsigned int i=1; i<words.size(); i++) {
      inputs_.back().push_back(std::stof(words[i]) * one_over_255);
    }
    numberOfExamples_++;
    
  }
  file.close();
*/  
}

unsigned int Database::getNumberOfExamples() {
  return numberOfExamples_;
}

std::vector<float> Database::getInput(unsigned int index) {
  if (index >= numberOfExamples_) {
    throw std::invalid_argument(
      "Index exeeds number of examples."
    );
  }
  return inputs_[index];
}

std::vector<float> Database::getOutput(unsigned int index) {
  if (index >= numberOfExamples_) {
    throw std::invalid_argument(
      "Index exeeds number of examples."
    );
  }
  return outputs_[index];
}

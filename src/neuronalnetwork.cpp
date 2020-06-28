#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include "utils.h"
#include "Neuron.h"
#include "Network.h"
#include "Database.h"
#include "Trainer.h"


int main(int argc, char** argv) {

  if ((argc == 4) && (std::string(argv[1]) == "exportpython")) {
    // example:
    //          neuronalnetwork exportpython /tmp/nw.txt /tmp/network.py
    //                 0              1            2           3
    std::string infilename = std::string(argv[2]);
    std::string outfilename = std::string(argv[3]);
    Network network;
    network.readFromFile(infilename);
    network.writePythonScript(outfilename);
  }


  if ((argc > 3) && (std::string(argv[1]) == "create")) {
    // example:
    //          neuronalnetwork create /tmp/nw.txt 784 16 16 10
    //               0            1         2       3   4  5  6
    std::string filename = std::string(argv[2]);
    std::vector< unsigned int > layers{};
    for(unsigned int i=3; i<argc; i++) {
      layers.push_back(std::stoi(argv[i]));
    }
    Network network(layers);
    network.writeToFile(filename);
  }

  if ((argc == 4) && (std::string(argv[1]) == "randomize") ) {
    // example:
    //          neuronalnetwork randomize /tmp/nw.txt 1.0
    //                 0            1          2       3
    std::string infilename = std::string(argv[2]);
    float factor = std::stof(argv[3]);
    std::string outfilename = std::string(argv[2]);
    Network network;
    network.readFromFile(infilename);
    network.randomizeBiases(factor);
    network.randomizeWeights(factor);
    network.writeToFile(outfilename);
  }

  if ((argc == 9) && (std::string(argv[1]) == "stochasticgradient") ) {
    // example:
    //           neuronalnetwork stochasticgradient /tmp/nw.txt ../database/database.txt 100 30 10000 0.01 10
    //                 0               1                2                    3            4   5   6    7    8
    //  2 : network filename
    //  3 : database filename
    //  4 : number of examples
    //  5 : number of cycles per dataset
    //  6 : number of examples to exclude at the end of the database for later test
    //  7 : factor for applying the gradient
    //  8 : number of datasets to be used for training

    std::string networkfilename(argv[2]);
    std::string databasefilename(argv[3]);
    unsigned int numberOfExamples = std::stoi(argv[4]);
    unsigned int numberOfCyclesPerDataset = std::stoi(argv[5]);
    unsigned int numberOfExamplesToExclude = std::stoi(argv[6]);
    float factor = std::stof(argv[7]);
    unsigned int numberOfDatasets = std::stoi(argv[8]);
    Network network;
    network.readFromFile(networkfilename);

    Trainer trainer(databasefilename);
    for(unsigned int i=0; i<numberOfDatasets; i++) {
      trainer.StochasticGradientOneBatch(
        network,
        numberOfExamples,
        numberOfCyclesPerDataset,
        numberOfExamplesToExclude,
        factor
      );
      network.writeToFile(networkfilename);
    }
  }

/*
//  Network network(std::vector<unsigned int> {28*28, 16, 16, 10});
//  network.randomizeBiases(1.0);
//  network.randomizeWeights(1.0);
  Network network;
  network.readFromFile("/tmp/nw_newtest.txt");

  std::default_random_engine generator(time(0));
  std::uniform_int_distribution<int> distribution(0, 99);


  Trainer trainer;
  Mnist mnist;
  mnist.readFromFile("../../mnist_recognition/mnist_train.csv");
 for(unsigned int k=0; k<1000; k++) {
    std::vector<unsigned int> indices{};
    for(unsigned int i=0; i<10000; i++) {
      indices.push_back(i);//distribution(generator));
    }
    float cost_old = trainer.calculate_cost(indices, network);
    float cost = cost_old;
    float last_cost = cost;
    network.resetDelta();
    float factor = 0.0001;
//    while ( cost > cost_old - 1.0) {
      for(unsigned int i=0; i<10000; i++) {
        network.reset();
        network.BackPropagation(mnist.getInput(i), mnist.getOutput(i));
        network.addGradientsToDelta();
      }
      network.applyDelta(-factor);
//      last_cost = cost;
//      cost = trainer.calculate_cost(indices, network);
//      if (last_cost < cost) {
//        factor *= 0.5;
//        std::cout << "                     factor = " << factor << '\n';
//        if (factor < 0.000001) {
//          cost_old = 1000.0;
//        }
//      }
//      std::cout << cost << '\n';
 //   }
    std::cout << "cycle " << k << ": " << trainer.calculate_cost(indices, network) << "    " << trainer.number_of_correct(indices, network) << '\n';
  network.writeToFile("/tmp/nw_newtest.txt");
  }
*/

/*
//  Network network(std::vector<unsigned int> {28*28, 32, 32, 10});
  Network network;
  network.readFromFile("/tmp/nw1234.txt");

  Trainer trainer;

  for(unsigned int i=0; i < 42; i++) {
//    trainer.reduce(network, 100);
  }

  for(unsigned int cycle = 0; cycle < 400; cycle++) {
    for(unsigned int i; i< network.getNumberOfNeurons(); i++) {
      network.getNeuronByIndex(i)->reduceWeights();
    }
    trainer.train(network, 1, 20, 0.5);
  }

  network.writeToFile("/tmp/nw1234.txt");
*/
/*
  Network parent1;
  parent1.readFromFile("/tmp/nw12.txt");
  Network parent2;
  parent2.readFromFile("/tmp/nw34.txt");
  Network child(parent1, parent2);
  child.writeToFile("/tmp/nw1234.txt");
*/
}


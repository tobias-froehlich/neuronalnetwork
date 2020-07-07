#include <iostream>
#include <gtest/gtest.h>
#include <random>
#include "../src/utils.h"
#include "../src/Neuron.h"
#include "../src/Network.h"
#include "../src/Database.h"

TEST(utils, activation_function) {
  float output = 0.0;
  output = utils::activation_function(-100.0);
  output = utils::activation_function(100.0);
  output = utils::activation_function(-1.0);
  output = utils::activation_function(1.0);
  output = utils::activation_function(0.0);
  output = utils::activation_function(-0.5);
  output = utils::activation_function(0.5);
}

TEST (utils, index_at_max) {
  ASSERT_EQ(utils::index_at_max(std::vector<float> {0, 1, 3, 2}), 2);
  ASSERT_EQ(utils::index_at_max(std::vector<float> {4, 1, 3, 2}), 0);
  ASSERT_EQ(utils::index_at_max(std::vector<float> {4, 1, 3, 5}), 3);
  ASSERT_EQ(utils::index_at_max(std::vector<float> {4}), 0);
}

TEST(Neuron, create_and_delete) {
  Neuron* neuron;
  neuron = new Neuron();
  delete neuron;
}

TEST(Neuron, forceOutput) {
  Neuron neuron;
  neuron.forceOutput(0.7);
  ASSERT_FLOAT_EQ(neuron.getOutput(), 0.7);
}

TEST(Neuron, adding_and_removing_sources) {
  Neuron neuron1;
  Neuron neuron2;
  Neuron neuron3;
  neuron2.addSource(&neuron1, 0.5);
  ASSERT_EQ(neuron1.getNumberOfSources(), 0);
  ASSERT_EQ(neuron2.getNumberOfSources(), 1);
  ASSERT_EQ(neuron1.getNumberOfDrains(), 1);
  ASSERT_EQ(neuron2.getNumberOfDrains(), 0);

  ASSERT_THROW(
    neuron2.getSource(1),
    std::invalid_argument
  );
  ASSERT_EQ(neuron2.getSource(0), &neuron1);
  ASSERT_EQ(neuron1.getDrain(0), &neuron2);
  ASSERT_THROW(
    neuron2.getWeight(1),
    std::invalid_argument
  );
  ASSERT_FLOAT_EQ(neuron2.getWeight(0), 0.5);

  ASSERT_THROW(
    neuron2.setWeight(5, 0.3),
    std::invalid_argument
  );
  neuron2.setWeight(0, 0.3);
  ASSERT_FLOAT_EQ(neuron2.getWeight(0), 0.3);

  ASSERT_THROW(
    neuron2.removeSourceByIndex(1),
    std::invalid_argument
  );

  neuron2.removeSourceByIndex(0);
  ASSERT_EQ(neuron2.getNumberOfSources(), 0);
  ASSERT_EQ(neuron1.getNumberOfDrains(), 0);

  neuron1.addSource(&neuron2, 0.5);
  ASSERT_EQ(neuron1.getNumberOfSources(), 1);
  neuron1.removeSource(&neuron3);
  ASSERT_EQ(neuron1.getNumberOfSources(), 1);
  neuron1.removeSource(&neuron2);
  ASSERT_EQ(neuron1.getNumberOfSources(), 0);
}

TEST (Neuron, calculating_output) {
  Neuron neuron1;
  Neuron neuron2;
  Neuron neuron3;

  neuron1.addSource(&neuron2, 0.5);
  neuron1.addSource(&neuron3, 0.7);
  neuron1.setBias(-0.3);
  neuron1.resetOutput();
  neuron2.resetOutput();
  neuron3.resetOutput();
  neuron2.forceOutput(0.1);
  neuron3.forceOutput(0.2);
  ASSERT_FLOAT_EQ(
    neuron1.getOutput(),
    utils::activation_function(
      0.1*0.5 + 0.2*0.7 - 0.3
    )
  );
  ASSERT_FLOAT_EQ(
    neuron1.getOutput(),
    utils::activation_function(
      0.1*0.5 + 0.2*0.7 - 0.3
    )
  );
}

TEST (Neuron, gradient1) {
  Neuron neuron1;
  neuron1.forceOutput(0.5);
  neuron1.calculateOutputGradientOutput(0.7);
  ASSERT_FLOAT_EQ(
    neuron1.getOutputGradient(),
    -0.4
  );  
}

TEST (Neuron, gradient2) {
  Neuron neuron1;
  Neuron neuron2;
  neuron2.addSource(&neuron1, 0.3);
  neuron1.setBias(-0.2);
  neuron2.setBias(-0.1);
  neuron1.forceOutput(0.5);
  neuron2.calculateOutputGradientOutput(0.7);
  ASSERT_FLOAT_EQ(
    neuron2.getOutputGradient(),
    2 * (utils::activation_function(0.5*0.3-0.1) - 0.7)
  );
}

TEST (Neuron, gradient3) {
  Neuron neuron1;
  Neuron neuron2;
  Neuron neuron3;
  neuron2.setBias(-0.5);
  neuron3.setBias(-0.1);
  neuron2.addSource(&neuron1, 0.2);
  neuron3.addSource(&neuron2, -0.3);
  neuron1.forceOutput(0.5);
  ASSERT_FLOAT_EQ(
    neuron3.getOutput(),
    utils::activation_function(utils::activation_function(0.5*0.2-0.5)*(-0.3)-0.1)
  );
  neuron1.resetOutput();
  neuron2.resetOutput();
  neuron3.resetOutput();
  neuron1.resetGradients();
  neuron2.resetGradients();
  neuron3.resetGradients();
  neuron1.resetDelta();
  neuron2.resetDelta();
  neuron3.resetDelta();

  for(unsigned int i=0; i<100; i++) {
    neuron1.resetOutput();
    neuron2.resetOutput();
    neuron3.resetOutput();
    neuron1.resetGradients();
    neuron2.resetGradients();
    neuron3.resetGradients();
    neuron1.resetDelta();
    neuron2.resetDelta();
    neuron3.resetDelta();
    neuron3.calculateOutputGradientOutput(0.5);
    neuron2.getWeightGradient(0);
    neuron3.getWeightGradient(0);
    neuron1.getBiasGradient();
    neuron2.getBiasGradient();
    neuron3.getBiasGradient();
    neuron1.addGradientToDelta();
    neuron2.addGradientToDelta();
    neuron3.addGradientToDelta();
    neuron1.applyDelta(-1.0);
    neuron2.applyDelta(-1.0);
    neuron3.applyDelta(-1.0);
  }
  
  ASSERT_FLOAT_EQ(neuron3.getOutput(), 0.5);
}

TEST (Network, create_and_delete) {
  Network* network = new Network();
  delete network;

  Network* network2 = new Network(
    std::vector<unsigned int> {4, 3, 2}
  );

  ASSERT_EQ(network2->getNumberOfNeurons(), 9);
  ASSERT_EQ(network2->getNumberOfInputs(), 4);
  ASSERT_EQ(network2->getNumberOfOutputs(), 2);
  ASSERT_EQ(network2->getNeuronByIndex(4)->getNumberOfSources(), 4);
  ASSERT_EQ(network2->getNeuronByIndex(6)->getNumberOfSources(), 4);
  ASSERT_EQ(network2->getNeuronByIndex(7)->getNumberOfSources(), 3);
  ASSERT_EQ(network2->getNeuronByIndex(8)->getNumberOfSources(), 3);

  delete network2;

  Network network3(
    std::vector<unsigned int> {4, 3, 2}
  );
  network3.randomizeBiases(1.0);
  network3.randomizeWeights(1.0);
  Network network4(network3);
  ASSERT_EQ(network4.getNumberOfNeurons(), 9);
  ASSERT_EQ(network4.getNumberOfInputs(), 4);
  ASSERT_EQ(network4.getNumberOfOutputs(), 2);
  ASSERT_EQ(network4.getNeuronByIndex(4)->getNumberOfSources(), 4);
  ASSERT_EQ(network4.getNeuronByIndex(6)->getNumberOfSources(), 4);
  ASSERT_EQ(network4.getNeuronByIndex(7)->getNumberOfSources(), 3);
  ASSERT_EQ(network4.getNeuronByIndex(8)->getNumberOfSources(), 3);
  std::vector<float> input {0.1, -0.2, -0.3, 0.4};
  network3.resetOutputs();
  network4.resetOutputs();
  network3.setInput(input);
  network4.setInput(input);
  std::vector<float> output3 = network3.getOutput();
  std::vector<float> output4 = network4.getOutput();
  for(unsigned int i=0; i<2; i++) {
    ASSERT_FLOAT_EQ(output3[i], output4[i]);
  } 
}

TEST (Network, create_and_remove_neuron) {
  Network network;
  ASSERT_EQ(network.getNumberOfNeurons(), 0);
  network.createNeuron();
  Neuron* neuron = network.getNeuronByIndex(0);
  ASSERT_EQ(network.getNumberOfNeurons(), 1);
  ASSERT_THROW(
    network.getNeuronByIndex(1),
    std::invalid_argument
  );
  ASSERT_EQ(network.getNeuronByIndex(0), neuron);
  ASSERT_THROW(
    network.removeNeuronByIndex(1),
    std::invalid_argument
  );
  Neuron* neuron2 = new Neuron();
  ASSERT_THROW(
    network.getIndexByNeuron(neuron2),
    std::invalid_argument
  );
  ASSERT_EQ(network.getIndexByNeuron(neuron), 0);
  network.removeNeuronByIndex(0);
  ASSERT_EQ(network.getNumberOfNeurons(), 0);

  delete neuron2;
}

TEST ( Network, create_connect_and_delete_neurons) {
  Network network;
  network.createNeuron();
  network.createNeuron();
  network.createNeuron();
  ASSERT_EQ(network.getNumberOfNeurons(), 3); 
  network.connect(0, 1, 0.5);
  ASSERT_EQ(network.getNeuronByIndex(0)->getNumberOfSources(), 0);
  ASSERT_EQ(network.getNeuronByIndex(1)->getNumberOfSources(), 1);

  network.resetOutputs();
  network.getNeuronByIndex(0)->forceOutput(0.1);
  ASSERT_FLOAT_EQ(network.getNeuronByIndex(1)->getOutput(), utils::activation_function(0.1*0.5));

  network.removeNeuronByIndex(0);
  ASSERT_EQ(network.getNumberOfNeurons(), 2);
  ASSERT_EQ(network.getNeuronByIndex(0)->getNumberOfSources(), 0);
}



TEST ( Network, create_and_delete_inputs_and_outputs) {
  Network network;
  network.createNeuron();
  network.createNeuron();
  network.createNeuron();
  network.createNeuron();
  

  network.connect(0, 1, 0.5);
  network.connect(1, 2, 0.2);

  network.getNeuronByIndex(1)->setBias(-0.4);
  network.getNeuronByIndex(2)->setBias(0.1);



  ASSERT_EQ(network.getNumberOfInputs(), 0);
  network.addInput(network.getNeuronByIndex(0));
  ASSERT_EQ(network.getNumberOfInputs(), 1);
  ASSERT_TRUE(network.isInput(network.getNeuronByIndex(0)));
  ASSERT_FALSE(network.isInput(network.getNeuronByIndex(1)));
  ASSERT_THROW(
    network.getInputByIndex(1),
    std::invalid_argument
  );
  ASSERT_EQ(
    network.getInputByIndex(0),
    network.getNeuronByIndex(0)
  );

  ASSERT_EQ(network.getNumberOfOutputs(), 0);
  network.addOutput(network.getNeuronByIndex(2));
  ASSERT_EQ(network.getNumberOfOutputs(), 1);
  ASSERT_TRUE(network.isOutput(network.getNeuronByIndex(2)));
  ASSERT_FALSE(network.isOutput(network.getNeuronByIndex(1)));
  ASSERT_THROW(
    network.getOutputByIndex(1),
    std::invalid_argument
  );
  ASSERT_EQ(
    network.getOutputByIndex(0),
    network.getNeuronByIndex(2)
  );
  network.resetOutputs();
  ASSERT_THROW(
    network.setInput(std::vector<float> {}),
    std::invalid_argument
  );
  
 
  ASSERT_THROW(
    network.setInput(std::vector<float> {0.4, 0.3}),
    std::invalid_argument
  );
  network.setInput(std::vector<float> {0.4});
  ASSERT_FLOAT_EQ(
    network.getNeuronByIndex(0)->getOutput(),
    0.4
  );
  std::vector<float> correct_output = {utils::activation_function(utils::activation_function(0.5*0.4-0.4)*0.2 + 0.1)};
  std::vector<float> output = network.getOutput();
  ASSERT_EQ(output.size(), correct_output.size());
  ASSERT_FLOAT_EQ(output[0], correct_output[0]);


  network.removeNeuronByIndex(0);
  ASSERT_EQ(network.getNumberOfInputs(), 0);
  network.removeNeuronByIndex(1);
  ASSERT_EQ(network.getNumberOfOutputs(), 0);
  
}

TEST (Network, read_from_file) {
  Network network;
  ASSERT_THROW(
    network.readFromFile("not_exists/file.txt"),
    std::invalid_argument
  );
  network.readFromFile("../test/testfiles/test1.txt");
  ASSERT_EQ(network.getNumberOfNeurons(), 4);
  ASSERT_EQ(network.getNumberOfInputs(), 2);
  ASSERT_EQ(network.getNumberOfOutputs(), 1);
  ASSERT_FLOAT_EQ(network.getNeuronByIndex(0)->getBias(), 0.0);
  ASSERT_FLOAT_EQ(network.getNeuronByIndex(1)->getBias(), 1.1);
  ASSERT_FLOAT_EQ(network.getNeuronByIndex(2)->getBias(), 1.2);
  ASSERT_FLOAT_EQ(network.getNeuronByIndex(3)->getBias(), 1.3);
  ASSERT_EQ(network.getNeuronByIndex(0)->getNumberOfSources(), 0);
  ASSERT_EQ(network.getNeuronByIndex(1)->getNumberOfSources(), 0);
  ASSERT_EQ(network.getNeuronByIndex(2)->getNumberOfSources(), 2);
  ASSERT_EQ(network.getNeuronByIndex(3)->getNumberOfSources(), 1);
  ASSERT_EQ(network.getNeuronByIndex(2)->getSource(0), network.getNeuronByIndex(0));
  ASSERT_EQ(network.getNeuronByIndex(2)->getSource(1), network.getNeuronByIndex(1));
  ASSERT_EQ(network.getNeuronByIndex(3)->getSource(0), network.getNeuronByIndex(2));
  ASSERT_FLOAT_EQ(network.getNeuronByIndex(2)->getWeight(0), 0.1);
  ASSERT_FLOAT_EQ(network.getNeuronByIndex(2)->getWeight(1), 0.2);
  ASSERT_FLOAT_EQ(network.getNeuronByIndex(3)->getWeight(0), 0.3);
}

TEST (Network, write_to_file) {
  Network network;

  network.createNeuron();
  network.createNeuron();
  network.createNeuron();
  network.createNeuron();

  network.addInput(network.getNeuronByIndex(0));
  network.addInput(network.getNeuronByIndex(1));
  network.addOutput(network.getNeuronByIndex(3));

  network.getNeuronByIndex(1)->setBias(1.1);
  network.getNeuronByIndex(2)->setBias(1.2);
  network.getNeuronByIndex(3)->setBias(1.3);

  network.connect(0, 2, 0.1);
  network.connect(1, 2, 0.2);
  network.connect(2, 3, 0.3);

  ASSERT_THROW(
    network.writeToFile("not_exsists/file.txt"),
    std::invalid_argument
  );
  network.writeToFile("../test/testfiles/test1.txt");
}
/*
TEST (Network, create_from_parents) {
  Network network1;
  Network network2;
  Network network3(std::vector<unsigned int>{2, 3, 2});
  Network network4(std::vector<unsigned int>{4, 3, 4});

  network1.readFromFile(
    "../test/testfiles/parent1.txt"
  );
  network2.readFromFile(
    "../test/testfiles/parent2.txt"
  );

  ASSERT_THROW(
    Network child(network1, network3),
    std::invalid_argument
  );

  ASSERT_THROW(
    Network child(network1, network4),
    std::invalid_argument
  );

  Network child(network1, network2);

  ASSERT_EQ(child.getNumberOfNeurons(), 12);
  ASSERT_EQ(child.getNumberOfInputs(), 3);
  ASSERT_EQ(child.getNumberOfOutputs(), 2);
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(3)->getBias(),
    0.3
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(6)->getBias(),
    0.6
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(7)->getBias(),
    0.3
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(9)->getBias(),
    0.5
  );
  ASSERT_EQ(
    child.getNeuronByIndex(3)->getNumberOfSources(),
    2
  );
  ASSERT_EQ(
    child.getNeuronByIndex(3)->getSource(0),
    child.getNeuronByIndex(0)
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(3)->getWeight(0),
    0.1
  ); 
  ASSERT_EQ(
    child.getNeuronByIndex(6)->getNumberOfSources(),
    1
  );
  ASSERT_EQ(
    child.getNeuronByIndex(6)->getSource(0),
    child.getNeuronByIndex(4)
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(6)->getWeight(0),
    0.7
  );
  ASSERT_EQ(
    child.getNeuronByIndex(7)->getNumberOfSources(),
    3
  );
  ASSERT_EQ(
    child.getNeuronByIndex(7)->getSource(0),
    child.getNeuronByIndex(0)
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(7)->getWeight(0),
    0.1
  );
  ASSERT_EQ(
    child.getNeuronByIndex(9)->getNumberOfSources(),
    1
  );
  ASSERT_EQ(
    child.getNeuronByIndex(9)->getSource(0),
    child.getNeuronByIndex(7)
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(9)->getWeight(0),
    0.5
  );
  ASSERT_EQ(
    child.getNeuronByIndex(10)->getNumberOfSources(),
    4
  );
  ASSERT_EQ(
    child.getNeuronByIndex(10)->getSource(0),
    child.getNeuronByIndex(5)
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(10)->getWeight(0),
    0.0
  );
  ASSERT_EQ(
    child.getNeuronByIndex(10)->getSource(3),
    child.getNeuronByIndex(9)
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(10)->getWeight(3),
    0.0
  );
  ASSERT_EQ(
    child.getNeuronByIndex(11)->getNumberOfSources(),
    4
  );
  ASSERT_EQ(
    child.getNeuronByIndex(11)->getSource(0),
    child.getNeuronByIndex(5)
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(11)->getWeight(0),
    0.0
  );
  ASSERT_EQ(
    child.getNeuronByIndex(11)->getSource(3),
    child.getNeuronByIndex(9)
  );
  ASSERT_FLOAT_EQ(
    child.getNeuronByIndex(11)->getWeight(3),
    0.0
  ); 
}
*/

TEST (Network, packpropagation) {
  Network network;
  network.readFromFile("../test/testfiles/parent1.txt");
  std::vector< std::vector<float> > inputs{{0.1, 0.2, 0.3}, {0.2, 0.7, 0.3}};
  std::vector< std::vector<float> > outputs{{0.2, 0.8}, {0.5, 0.3}};
  std::cout << "---------\n";
  for(unsigned int k=0; k<1000; k++) {
    network.resetDelta();
    for(unsigned int i=0; i<inputs.size(); i++) {
      network.resetOutputs();
      network.BackPropagation(inputs[i], outputs[i]);
      network.addGradientsToDelta();
    }
    network.applyDelta(-0.1);
    float cost = 0.0;
    for(unsigned int i=0; i<inputs.size(); i++) {
      for(unsigned int j=0; j<outputs[i].size(); j++) {
        cost += utils::square(network.getOutput()[j] - outputs[i][j]);
      }
    }

    std::cout << cost << '\n';
  }
  network.writeToFile("../test/testfiles/packprop_output.txt");
}

TEST (Database, create_and_delete) {
  Database* database = new Database();
  delete database;
}

TEST (Database, readFromFile) {
  Database database;
  ASSERT_THROW(
    database.readFromFile("doesnotexist.txt"),
    std::invalid_argument
  );
  database.readFromFile("../test/testfiles/database.txt");

  ASSERT_THROW(
    database.getInput(60000),
    std::invalid_argument
  );
  ASSERT_THROW(
    database.getOutput(60000),
    std::invalid_argument
  );
  std::vector<float> input;
  std::vector<float> output;
  
  input = database.getInput(4);
  output = database.getOutput(4);
  ASSERT_EQ(output.size(), 28*28);
  ASSERT_FLOAT_EQ(output[0], 0.0);
  ASSERT_FLOAT_EQ(output[123], 0.42);
  ASSERT_EQ(input.size(), 28*28);
  ASSERT_FLOAT_EQ(input[0], 0.0);
  ASSERT_FLOAT_EQ(input[208], 0.22);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

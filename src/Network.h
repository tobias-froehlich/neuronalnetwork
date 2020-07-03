#ifndef NETWORK_H_
#define NETWORK_H_

class Network {
  private:
    std::vector< Neuron* > neurons_{};
    std::vector< Neuron* > inputs_{};
    std::vector< Neuron* > outputs_{};
    unsigned int numberOfNeurons_ = 0;
    unsigned int numberOfInputs_ = 0;
    unsigned int numberOfOutputs_ = 0;
    std::default_random_engine generator_;
    std::uniform_real_distribution<float> distribution_{-1.0, 1.0};
  public:
    Network();
    Network(const Network & other);
    Network(std::vector<unsigned int> layers);
    Network(const Network & parent1, const Network & parent2);
    ~Network();
    void passParameters(Network & other) const;
    unsigned int getNumberOfNeurons() const;
    unsigned int getNumberOfInputs() const;
    unsigned int getNumberOfOutputs() const;
    Neuron* getNeuronByIndex(unsigned int index) const;
    unsigned int getIndexByNeuron(Neuron* neuron) const;
    bool isInput(Neuron* neuron) const;
    bool isOutput(Neuron* neuron) const;
    void createNeuron();
    void addInput(Neuron* neuron);
    Neuron* getInputByIndex(unsigned int index) const;
    void addOutput(Neuron* neuron);
    Neuron* getOutputByIndex(unsigned int index) const;
    void removeNeuronByIndex(unsigned int index);
    void connect(
      unsigned int index_from,
      unsigned int index_to,
      float weight
    );
    void reset();
    void setInput(std::vector<float> input);
    std::vector<float> getOutput();
    void readFromFile(std::string filename);
    void writeToFile(std::string filename) const;
    void randomizeWeights(float factor);
    void randomizeBiases(float factor);
    void randomizeOutputWeights(float factor);
    void randomizeOutputBiases(float factor);
    void resetDelta();
    void resetGradients();
    void addGradientsToDelta();
    float absDelta();
    void applyDelta(float factor);
    void applyDeltaBiasesOnly(float factor);
    void applyDeltaWeightsOnly(float factor);
    void BackPropagation(std::vector<float> input, std::vector<float> output);
    void writePythonScript(std::string filename);
};

#endif

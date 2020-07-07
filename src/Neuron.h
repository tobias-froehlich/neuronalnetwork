#ifndef NEURON_H_
#define NEURON_H_

class Neuron {
  private:
    unsigned int number_of_sources_ = 0;
    unsigned int number_of_drains_ = 0;
    std::vector<Neuron*> source_neurons_ {};
    std::vector<float> weights_ {};
    std::vector<float> weightsGradient_ {};
    std::vector<float> weightsDelta_ {};
    std::vector<Neuron*> drain_neurons_ {};
    std::vector<unsigned int> drainSourceIndices_{};
    float bias_ = 0.0;
    float biasGradient_ = 0.0;
    float biasDelta_ = 0.0;
    float outputWithoutActivationFunction_ = 0.0;
    float output_ = 0.0;
    float outputGradient_ = 0.0;
    bool set_ = false;
    bool weightGradientSet_ = false;
    bool biasGradientSet_ = false;
    bool outputGradientSet_ = false;
    std::string pythonVarName_{""};
  public:
    Neuron();
    ~Neuron();
    unsigned int getNumberOfSources();
    unsigned int getNumberOfDrains();
    Neuron* getSource(unsigned int index);
    Neuron* getDrain(unsigned int index);
    unsigned int getDrainSourceIndex(unsigned int index);
    void setWeight(unsigned int index, float weight);
    float getWeight(unsigned int index);
    void addSource(Neuron* neuron, float weight);
    void addDrain(Neuron* neuron, float weight);
    void removeSourceByIndex(unsigned int index);
    void removeSource(Neuron* neuron);
    void removeDrainByIndex(unsigned int index);
    void removeDrain(Neuron* neuron);
    void forceOutput(float output);
    float getPreviousOutput();
    float getOutput();
    float getOutputWithoutActivationFunction();
    void reset();
    void setBias(float bias);
    float getBias();
    void resetGradients();
    void resetDelta();
    void calculateOutputGradientOutput(float correct_output);
    float getWeightGradient(unsigned int index);
    float getBiasGradient();
    float getOutputGradient();
    void addGradientToDelta();
    float sumOfDeltaSquared();
    void applyDelta(float factor);
    void applyDeltaBiasOnly(float factor);
    void applyDeltaWeightsOnly(float factor);
    void reduceWeights();
    void setPythonVarName(std::string name);
    std::string getPythonVarName();
    void getPythonScript(std::ostream & os);
};

#endif

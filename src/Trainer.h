#ifndef TRAINER_H_
#define TRAINER_H_

class Trainer {
  private:
    Database database_;
  public:
    Trainer(std::string databasefilename);
    ~Trainer();
    unsigned int number_of_correct(std::vector<unsigned int> indices, Network & network);
    float calculate_cost(std::vector<unsigned int> indices, Network & network);
    void train(Network & network, unsigned int number_of_cycles, unsigned int number_of_samples, float randomizefactor);
    void trainOutputOnly(Network & network, unsigned int number_of_cycles, unsigned int number_of_samples, float randomizefactor);
    void reduce(Network & network, unsigned int number_of_samples);
    void StochasticGradientOneBatch(Network & network, unsigned int numberOfSamples, unsigned int numberOfCycles, unsigned int without_last, float factor);
};
#endif

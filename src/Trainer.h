#ifndef TRAINER_H_
#define TRAINER_H_

class Trainer {
  private:
    Database database_;
  public:
    Trainer(std::string databasefilename);
    ~Trainer();
    float calculate_cost(std::vector<unsigned int> indices, Network & network);
    void StochasticGradientOneBatch(Network & network, unsigned int numberOfSamples, unsigned int numberOfCycles, unsigned int without_last, float factor);
};
#endif

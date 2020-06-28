#ifndef DATABASE_H_
#define DATABASE_H_

class Database {
  private:
    unsigned int numberOfExamples_ = 0;
    std::vector<int> numbers_{};
    std::vector< std::vector<float> > outputs_{};
    std::vector< std::vector<float> > inputs_{};
  public:
    void readFromFile(std::string filename);
    unsigned int getNumberOfExamples();
    std::vector<float> getInput(unsigned int index);
    std::vector<float> getOutput(unsigned int index);
};

#endif

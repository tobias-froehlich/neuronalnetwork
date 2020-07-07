#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <ctime>
#include "utils.h"

namespace utils {

  float activation_function(float input) {
    return 0.5 * tanh(input - 0.5) + 0.5;
/*    if (input < 0.0) {
      return input * 0.05;
    }
    else if (input < 1.0) {
      return input;
    }
    else {
      return 1.0 + (input - 1.0)* 0.05;
    }*/
  }

  float activation_function_derivative(float input) {
    return 0.5 * (1.0 - square(tanh(input - 0.5)));
/*    if (input < 0.0) {
      return 0.05;
    }
    else if (input < 1.0) {
      return 1.0;
    }
    else {
      return 0.05;
    }*/
  }

  std::vector< std::string > split(std::string str, char delimiter) {
      std::vector< std::string > words{};
      std::string word = "";
      for (char& c : str) {
        if (c != delimiter) {
          word.append(1, c);
        }
        else if (word.size() > 0) {
          words.push_back(word);
          word = "";
        }
      }
      if (word.size() > 0) {
        words.push_back(word);
        word = "";
      }
      return words;
  }

  unsigned int index_at_max(std::vector<float> v) {
    float maximum = v[0];
    unsigned int result = 0;
    for(unsigned int i=1; i<v.size(); i++) {
      if (v[i] > maximum) {
        maximum = v[i];
        result = i;
      }
    }
    return result;
  }

  float square(float x) {
    return x*x;
  }

  void log(std::string text) {
    std::ofstream file;
    file.open("/tmp/log.txt", std::ios_base::app);
    file << time(0) << " ";
    file << text;
    file << '\n';
    std::cout << text << '\n';
    file.close();
  }
}

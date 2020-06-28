#ifndef UTILS_H_
#define UTILS_H_

namespace utils {
  float activation_function(float input);
  float activation_function_derivative(float input);
  std::vector< std::string > split(std::string str, char delimiter);
  unsigned int index_at_max(std::vector<float> v);
  float square(float x);
  void log(std::string text);
}

#endif

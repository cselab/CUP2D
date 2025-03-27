#ifndef CubismUP_2D_FactoryFileLineParser_h
#define CubismUP_2D_FactoryFileLineParser_h
#include "Cubism/ArgumentParser.h"
#include <algorithm>
#include <locale>
#include <sstream>
#include <utility>
class FactoryFileLineParser : public cubism::ArgumentParser {
protected:
  static std::string trim(std::string str) {
    size_t i = 0, j = str.length();
    while (i < j && isspace(str[i]))
      i++;
    while (j > i && isspace(str[j - 1]))
      j--;
    return str.substr(i, j - i);
  }
public:
  FactoryFileLineParser(std::istringstream &is_line)
      : cubism::ArgumentParser(0, NULL, '#') {
    std::string key, value;
    while (std::getline(is_line, key, '=')) {
      if (std::getline(is_line, value, ' ')) {
	mapArguments[trim(key)] = cubism::Value(trim(value));
      }
    }
    mute();
  }
};
#endif

//
//  CubismUP_2D
//  Copyright (c) 2021 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//

#ifndef CubismUP_2D_FactoryFileLineParser_h
#define CubismUP_2D_FactoryFileLineParser_h

#include "Cubism/ArgumentParser.h"

#include <algorithm>
#include <locale>
#include <sstream>
#include <utility>

class FactoryFileLineParser: public cubism::ArgumentParser
{
protected:
   std::string trim(std::string str) {
        size_t i = 0, j = str.length();
        while (i < j && isspace(str[i]))
            i++;
        while (j > i && isspace(str[j - 1]))
            j--;
        return str.substr(i, j - i);
  }
public:

    FactoryFileLineParser(std::istringstream & is_line)
    : cubism::ArgumentParser(0, NULL, '#') // last char is comment leader
    {
        std::string key,value;
        while( std::getline(is_line, key, '=') )
        {
            if( std::getline(is_line, value, ' ') )
            {
                // add "-" because then we can use the same code for parsing factory as command lines
                //mapArguments["-"+trim(key)] = Value(trim(value));
                mapArguments[trim(key)] = cubism::Value(trim(value));
            }
        }

        mute();
    }
};

#endif // CubismUP_2D_FactoryFileLineParser_h

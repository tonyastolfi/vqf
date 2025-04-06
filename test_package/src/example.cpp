#include "vqf/vqf.hpp"
#include "vqf/vqf_filter.h"

#include <iostream>

int main()
{
   if (vqf::entry_point() && vqf_init<8>(100) != nullptr) {
      std::cout << "Test package is working!" << std::endl;
   }
   return 0;
}

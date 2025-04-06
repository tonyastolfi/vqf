#include "vqf/vqf.hpp"
#include "vqf/vqf_filter.h"
#include "vqf/vqf_wrapper.h"

#include <iostream>

int main()
{
   if (vqf::entry_point()) {
      std::cout << "Test package is working!" << std::endl;
   }
   return 0;
}

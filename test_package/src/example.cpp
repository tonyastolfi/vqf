#include "vqf/vqf.hpp"

#include <iostream>

int main()
{
    if (vqf::entry_point()) {
        std::cout << "Test package is working!" << std::endl;
    }
    return 0;
}

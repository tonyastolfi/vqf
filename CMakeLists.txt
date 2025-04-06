cmake_minimum_required(VERSION 3.20)
project(vqf C CXX)

include(${COR_CMAKE_INCLUDE_DIR}/common.cmake)
include(${CMAKE_BINARY_DIR}/conan_find_requirements.cmake)

add_compile_options(-msse4.2 -march=native -m64 -frename-registers -fgnu-tm -DENABLE_THREADS)

add_subdirectory(src)

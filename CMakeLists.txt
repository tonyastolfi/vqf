cmake_minimum_required(VERSION 3.20)
project(vqf C CXX)

include(${COR_CMAKE_INCLUDE_DIR}/common.cmake)
include(${CMAKE_BINARY_DIR}/conan_find_requirements.cmake)

#==#==========+==+=+=++=+++++++++++-+-+--+----- --- -- -  -  -   -
# Disable link-time optimization if the compilation flags are not compatible.
#
if ("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
    MATCHES "-fsanitize=address|-fno-lto")
  message("\nNDTE: Disabling LTO because of compile flags: ${CMAKE_MATCH_0}\n")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
else ()
  set(VQF_EXTRA_COMPILE_FLAGS -fgnu-tm)
endif ()

add_compile_options(-msse4.2 -march=native -m64 -frename-registers ${VQF_EXTRA_COMPILE_FLAGS})

add_subdirectory(src)

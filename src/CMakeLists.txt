cmake_minimum_required(VERSION 3.20)


file(GLOB_RECURSE vqf_Sources
  LIST_DIRECTORIES false
  ${CMAKE_CURRENT_SOURCE_DIR}/*.c*
  )

file(GLOB_RECURSE vqf_Headers
  LIST_DIRECTORIES false
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/*.i
  )

#+++++++++++-+-+--+----- --- -- -  -  -   -
# Find test source files and remove them from the sources list.
#
file(GLOB_RECURSE vqf_TestSources
  LIST_DIRECTORIES false
  ${CMAKE_CURRENT_SOURCE_DIR}/*.test.cpp
  )

foreach (_file "FORCE_LIST_NOT_EMPTY;${vqf_TestSources}")
  list(REMOVE_ITEM vqf_Sources ${_file})
endforeach ()

add_library(vqf ${vqf_Sources})

# Install the library, if we are not header only.
#
install(TARGETS vqf
  LIBRARY DESTINATION lib
  )

# Install each header in the correct relative directory.
#
foreach (_header ${vqf_Headers})
  cmake_path(RELATIVE_PATH _header)
  cmake_path(GET _header PARENT_PATH _destination)
  install(FILES ${_header} DESTINATION include/${_destination})
endforeach ()

add_executable(
  vqf_Test
  ${vqf_TestSources}
  )

target_link_libraries(
  vqf_Test
  PRIVATE vqf
  )

add_test(
  NAME vqf_Test
  COMMAND vqf_Test
  )

# Set library dependencies
#
target_link_libraries(
  vqf
  PUBLIC openssl::openssl
  #
  # Add other library -> library dependencies here.
  #
  )

target_link_libraries(
  vqf_Test
  PRIVATE gtest::gtest
  #
  # Add other test -> library dependencies here
  #
  )

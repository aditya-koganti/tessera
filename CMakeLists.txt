cmake_minimum_required(VERSION 3.16)
project(tessera VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TESSERA_BUILD_TESTS "Build unit and integration tests" ON)
option(TESSERA_WERROR "Treat warnings as errors" OFF)
option(TESSERA_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(tessera_warnings INTERFACE)
if(MSVC)
  target_compile_options(tessera_warnings INTERFACE /W4 $<$<BOOL:${TESSERA_WERROR}>:/WX>)
else()
  target_compile_options(tessera_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow
                         $<$<BOOL:${TESSERA_WERROR}>:-Werror>)
endif()

if(TESSERA_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(tessera
  src/systolic.cpp
  src/simulator.cpp
  src/compiler.cpp
  src/reference.cpp)
target_include_directories(tessera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tessera PRIVATE tessera_warnings)

add_executable(tessera-run tools/tessera_cli.cpp)
target_link_libraries(tessera-run PRIVATE tessera tessera_warnings)

if(TESSERA_BUILD_TESTS)
  enable_testing()
  find_package(GTest 1.10 CONFIG QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(tessera_tests
    tests/test_systolic.cpp
    tests/test_simulator.cpp
    tests/test_compiler.cpp)
  target_link_libraries(tessera_tests PRIVATE tessera tessera_warnings GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(tessera_tests DISCOVERY_TIMEOUT 60)

  # End-to-end tests drive the CLI from PyTest and check results independently in Python.
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pytest"
                    RESULT_VARIABLE TESSERA_NO_PYTEST OUTPUT_QUIET ERROR_QUIET)
    if(TESSERA_NO_PYTEST EQUAL 0)
      add_test(NAME pytest_cli
               COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)
      set_tests_properties(pytest_cli PROPERTIES
                           ENVIRONMENT "TESSERA_BIN=$<TARGET_FILE:tessera-run>")
    else()
      message(STATUS "pytest not found; skipping Python end-to-end tests")
    endif()
  endif()
endif()

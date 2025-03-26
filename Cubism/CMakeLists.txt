cmake_minimum_required(VERSION 3.9)
project(Cubism)

set(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(SRC_DIR  "${ROOT_DIR}/src")

set(COMMON_SOURCES
    "${SRC_DIR}/ArgumentParser.cpp"
)

add_library(Cubism STATIC ${COMMON_SOURCES})
add_library(Cubism::Cubism ALIAS Cubism)
target_include_directories(Cubism PUBLIC "${ROOT_DIR}/include")

# -fPIC is necessary when compiling as a shared object. Although it does affect
# the performance, we enable it for convenience always, as the only compiled
# file here is for initialization.
set_property(TARGET Cubism PROPERTY POSITION_INDEPENDENT_CODE 1)

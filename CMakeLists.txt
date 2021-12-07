cmake_minimum_required(VERSION 3.12)
project(CubismUP_2D)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Set Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Paths and target names.
set(BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(SRC_DIR "${ROOT_DIR}/source")
set(DEP_BUILD_DIR "${ROOT_DIR}/dependencies/build")

set(EXE "cubismup2d_simulation")
set(CORE "cubismup2d_core")
set(PRIVATEDEP "cubismup2d_privatedep")  # Private dependencies and flags.
set(PYLIB "libcubismup2d")


# Options.
option(CUP2D_BUILD_EXE "Build the CubismUP_2D executable" ON)
option(CUP2D_BUILD_PY "Build Python bindings" ON)
set(CUP2D_BLOCK_SIZE "8" CACHE STRING "Number of grid points in a block, per dimension")
set(CUP2D_ROOT_DIR "${ROOT_DIR}" CACHE INTERNAL "CubismUP_2D repository root path")
set(CUP2D_BUILD_DIR "${BUILD_DIR}" CACHE INTERNAL "CubismUP_2D build path")
set(CUP2D_PRECISION "double" CACHE STRING "floating point precision")
set_property(CACHE CUP2D_PRECISION PROPERTY STRINGS single double long_double)


### Dependencies.

# Cubism.
add_subdirectory("${ROOT_DIR}/Cubism")
# TODO: Add prefix to _BS_ and DIMENSION macros.
target_compile_definitions(Cubism PUBLIC CUBISM_ALIGNMENT=32)
target_compile_definitions(Cubism PUBLIC CUBISM_USE_HDF)
target_compile_definitions(Cubism PUBLIC DIMENSION=2)
target_compile_definitions(Cubism PUBLIC _BS_=${CUP2D_BLOCK_SIZE})

find_package(GSL REQUIRED)
find_package(HDF5 REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)


# Targets.
add_library(${PRIVATEDEP} INTERFACE)
set(_PRIVATE_FLAGS -Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized -fmax-errors=3)
set(_PRIVATE_RELEASE_FLAGS
    -march=native
    -fstrict-aliasing
    -ffast-math
    -falign-functions=32
    -ftree-vectorize
    -fmerge-all-constants
)
# TODO: figure out how to convince cmake to send -std=c++17 and not -std=gnu++17.
target_compile_features(${PRIVATEDEP} INTERFACE cxx_std_17)
target_compile_options(${PRIVATEDEP} INTERFACE
                       ${_PRIVATE_FLAGS}
                       $<$<CONFIG:Release>:${_PRIVATE_RELEASE_FLAGS}>)

add_library(
    ${CORE} STATIC EXCLUDE_FROM_ALL
    "${SRC_DIR}/Obstacles/CStartFish.cpp"
    "${SRC_DIR}/Obstacles/CarlingFish.cpp"
    "${SRC_DIR}/Obstacles/ExperimentFish.cpp"
    "${SRC_DIR}/Obstacles/Fish.cpp"
    "${SRC_DIR}/Obstacles/FishData.cpp"
    "${SRC_DIR}/Obstacles/Naca.cpp"
    "${SRC_DIR}/Obstacles/NeuroKinematicFish.cpp"
    "${SRC_DIR}/Obstacles/ShapeLibrary.cpp"
    "${SRC_DIR}/Obstacles/ShapesSimple.cpp"
    "${SRC_DIR}/Obstacles/SmartCylinder.cpp"
    "${SRC_DIR}/Obstacles/StefanFish.cpp"
    "${SRC_DIR}/Obstacles/Teardrop.cpp"
    "${SRC_DIR}/Obstacles/Waterturbine.cpp"
    "${SRC_DIR}/Obstacles/Windmill.cpp"
    "${SRC_DIR}/Obstacles/ZebraFish.cpp"
    "${SRC_DIR}/Operators/AdaptTheMesh.cpp"
    "${SRC_DIR}/Operators/ComputeForces.cpp"
    "${SRC_DIR}/Operators/Helpers.cpp"
    "${SRC_DIR}/Operators/PressureSingle.cpp"
    "${SRC_DIR}/Operators/PutObjectsOnGrid.cpp"
    "${SRC_DIR}/Operators/advDiff.cpp"
    "${SRC_DIR}/Poisson/AMRSolver.cpp"
    "${SRC_DIR}/Shape.cpp"
    "${SRC_DIR}/Simulation.cpp"
    "${SRC_DIR}/SimulationData.cpp"
    "${SRC_DIR}/Utils/BufferedLogger.cpp"
)

add_library(cubismup2d::core ALIAS ${CORE})
target_link_libraries(${CORE} PRIVATE ${PRIVATEDEP})
target_link_libraries(${CORE} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${CORE} PUBLIC MPI::MPI_CXX)
target_link_libraries(${CORE} PUBLIC Cubism::Cubism)
target_link_libraries(${CORE} PUBLIC GSL::gsl)
target_link_libraries(${CORE} PUBLIC ${HDF5_LIBRARIES})
target_include_directories(${CORE} PUBLIC ${HDF5_INCLUDE_DIRS})
set_property(TARGET ${CORE} PROPERTY POSITION_INDEPENDENT_CODE 1)

if ("${CUP2D_PRECISION}" STREQUAL float)
    target_compile_definitions(${CORE} PUBLIC _FLOAT_PRECISION_)
elseif ("${CUP2D_PRECISION}" STREQUAL double)
    target_compile_definitions(${CORE} PUBLIC _DOUBLE_PRECISION_)
elseif ("${CUP2D_PRECISION}" STREQUAL long_double)
    target_compile_definitions(${CORE} PUBLIC _LONG_DOUBLE_PRECISION_)
else()
    message(FATAL_ERROR "unrecognized CUP2D_PRECISION: ${CUP2D_PRECISION}")
endif()

if (CUP2D_BUILD_EXE)
    add_executable(${EXE} "${ROOT_DIR}/source/main.cpp")
    target_link_libraries(${EXE} PUBLIC ${CORE})
endif()

if (CUP2D_BUILD_PY)
    set(PYTHON_SOURCES
        ${SRC_DIR}/Bindings/Simulation.cpp
    )

    # Check if pybind11 was imported by a potential parent cmake file.
    if (TARGET pybind11::pybind11)
        # If so, directly use pybind11_add_module.
        pybind11_add_module(
            ${PYLIB}
            ${PYTHON_SOURCES})
    else()
        # Otherwise, find Python & pybind11 package, and use Python_add_library.
        find_package(Python COMPONENTS Interpreter Development)
        if (NOT ${Python_Interpreter_FOUND})
            message(FATAL_ERROR "Python not found.")
        endif()
        Python_add_library(
            ${PYLIB}
            MODULE
            WITH_SOABI
            ${PYTHON_SOURCES})
        execute_process(
            COMMAND ${Python_EXECUTABLE} -m pybind11 --includes
            OUTPUT_VARIABLE _PYBIND11_INCLUDES
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _RETCODE)
        if (NOT "${_RETCODE}" STREQUAL "0")
            message(FATAL_ERROR "pybind11 not installed, cannot compile Python bindings!")
        endif()
        separate_arguments(_PYBIND11_INCLUDES UNIX_COMMAND ${_PYBIND11_INCLUDES})
        target_include_directories(${PYLIB} PRIVATE ${_PYBIND11_INCLUDES})
    endif()
    target_link_libraries(${PYLIB} PUBLIC ${CORE})
    target_link_libraries(${PYLIB} PRIVATE ${PRIVATEDEP})
endif()

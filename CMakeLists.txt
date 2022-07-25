cmake_minimum_required(VERSION 3.12)

# Enable C because HDF5 is not found on some system otherwise.
project(CubismUP_2D LANGUAGES C CXX)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
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
option(CUP2D_BUILD_EXE "Build the CubismUP_2D executable" OFF)
option(CUP2D_BUILD_PY "Build Python bindings" ON)
option(CUP2D_BACKWARD_CPP "Use backward-cpp for stack trace" ON)
set(CUP2D_BLOCK_SIZE "8" CACHE STRING "Number of grid points in a block, per dimension")
set(CUP2D_ROOT_DIR "${ROOT_DIR}" CACHE INTERNAL "CubismUP_2D repository root path")
set(CUP2D_BUILD_DIR "${BUILD_DIR}" CACHE INTERNAL "CubismUP_2D build path")
set(CUP2D_PRECISION "double" CACHE STRING "floating point precision")
set_property(CACHE CUP2D_PRECISION PROPERTY STRINGS single double long_double)
if (CMAKE_CUDA_COMPILER)
    option(CUP2D_CUDA "Compile optional CUDA features" ON)
else()
    option(CUP2D_CUDA "Compile optional CUDA features" OFF)
    if (CUP2D_CUDA)
        message(FATAL_ERROR "CUDA compiler not found, cannot compile with CUP2D_CUDA enabled.")
    endif()
endif()


### Dependencies.

# backward-cpp
if (CUP2D_BACKWARD_CPP AND NOT TARGET backward)
    include(FetchContent)
    FetchContent_Declare(
        backward-cpp
        GIT_REPOSITORY https://github.com/bombela/backward-cpp
        GIT_TAG v1.6
        EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(backward-cpp)
endif()

# Cubism.
add_subdirectory("${ROOT_DIR}/Cubism")
# TODO: Add prefix to _BS_ and DIMENSION macros.
target_compile_definitions(Cubism PUBLIC CUBISM_ALIGNMENT=32)
target_compile_definitions(Cubism PUBLIC CUBISM_USE_MAP)
target_compile_definitions(Cubism PUBLIC DIMENSION=2)
target_compile_definitions(Cubism PUBLIC _BS_=${CUP2D_BLOCK_SIZE})

find_package(GSL REQUIRED)
find_package(HDF5 REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# On CRAY systems MPI_CXX_INCLUDE_DIRS are empty so we apply this workaround to
# make mpi.h available from CUDA files.
if (DEFINED ENV{CRAY_MPICH_DIR})
    set(MPI_C_INCLUDE_DIRS   "$ENV{CRAY_MPICH_DIR}/include")
    set(MPI_CXX_INCLUDE_DIRS "$ENV{CRAY_MPICH_DIR}/include")
endif()

# Targets.
add_library(${PRIVATEDEP} INTERFACE)
set(_PRIVATE_FLAGS -Wall -Wextra -Wno-unused-parameter -Wno-maybe-uninitialized -fmax-errors=3)
set(_PRIVATE_RELEASE_FLAGS
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler -march=native>
    -falign-functions=32
    -fmerge-all-constants
    -fstrict-aliasing
    -ftree-vectorize
)
target_compile_features(${PRIVATEDEP} INTERFACE cxx_std_17)
target_compile_options(${PRIVATEDEP} INTERFACE
                       ${_PRIVATE_FLAGS}
                       $<$<CONFIG:RelWithDebInfo>:${_PRIVATE_RELEASE_FLAGS}>
                       $<$<CONFIG:Release>:${_PRIVATE_RELEASE_FLAGS}>)
if (CUP2D_CUDA)
    find_package(CUDAToolkit)
    # Safest option is not to pass any architecture flags, in case the code is
    # compiled on machines different than those the code is executed on.
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
        set_property(TARGET ${PRIVATEDEP} PROPERTY CUDA_ARCHITECTURES OFF)
    endif()
    # cuda_std_17 might not work, manually add -std=c++17 instead.
    # target_compile_features(${PRIVATEDEP} INTERFACE cuda_std_17)
    target_compile_options(${PRIVATEDEP} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)
    target_compile_definitions(${PRIVATEDEP} INTERFACE -DGPU_POISSON)
    target_include_directories(${PRIVATEDEP} INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(${PRIVATEDEP} INTERFACE CUDA::cublas CUDA::cusparse)
endif()

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
    "${SRC_DIR}/Operators/advDiffSGS.cpp"
    "${SRC_DIR}/Operators/ImportExportUniform.cpp"
    "${SRC_DIR}/Operators/Forcing.cpp"
    "${SRC_DIR}/Poisson/AMRSolver.cpp"
    "${SRC_DIR}/Poisson/Base.cpp"
    "${SRC_DIR}/Shape.cpp"
    "${SRC_DIR}/Simulation.cpp"
    "${SRC_DIR}/SimulationData.cpp"
    "${SRC_DIR}/Utils/BufferedLogger.cpp"
    "${SRC_DIR}/Utils/StackTrace.cpp"
)
if (CUP2D_CUDA)
    target_sources(
        ${CORE}
        PRIVATE
        "${SRC_DIR}/Poisson/BiCGSTAB.cu"
        "${SRC_DIR}/Poisson/ExpAMRSolver.cpp"
        "${SRC_DIR}/Poisson/LocalSpMatDnVec.cpp"
    )
    target_include_directories(${CORE} PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    get_cmake_property(_variableNames VARIABLES)
endif()

add_library(cubismup2d::core ALIAS ${CORE})
target_compile_definitions(${CORE} PUBLIC CUP2D_BLOCK_SIZE=${CUP2D_BLOCK_SIZE})
target_link_libraries(${CORE} PRIVATE ${PRIVATEDEP})
target_link_libraries(${CORE} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${CORE} PUBLIC MPI::MPI_CXX)
target_link_libraries(${CORE} PUBLIC Cubism::Cubism)
target_link_libraries(${CORE} PUBLIC GSL::gsl)
target_link_libraries(${CORE} PUBLIC ${HDF5_LIBRARIES})
target_include_directories(${CORE} PUBLIC ${HDF5_INCLUDE_DIRS})
# target_compile_options(${CORE} PUBLIC -fsanitize=address -static-libasan)
# target_link_options(${CORE} PUBLIC -fsanitize=address -static-libasan)
set_property(TARGET ${CORE} PROPERTY CXX_EXTENSIONS OFF)
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

if (CUP2D_BACKWARD_CPP)
    add_backward(${CORE})
    target_compile_definitions(${CORE} PUBLIC CUP2D_BACKWARD_CPP)
    target_link_libraries(${CORE} PUBLIC ${BACKWARD_LIBRARIES})
endif()

if (CUP2D_BUILD_EXE)
    add_executable(${EXE} "${ROOT_DIR}/source/main.cpp")
    target_link_libraries(${EXE} PUBLIC ${CORE})
endif()

if (CUP2D_BUILD_PY)
    if (NOT TARGET pybind11::pybind11)
        find_package(Python COMPONENTS REQUIRED Interpreter Development)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE _PYBIND11_CMAKEDIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _RETCODE)
        if (NOT "${_RETCODE}" STREQUAL "0")
            message(FATAL_ERROR "pybind11 not installed, cannot compile Python bindings!")
        endif()
        find_package(pybind11 REQUIRED HINTS "${_PYBIND11_CMAKEDIR}")
    endif()

    pybind11_add_module(
        ${PYLIB}
        MODULE
        NO_EXTRAS
        "${SRC_DIR}/Bindings/Bindings.cpp"
        "${SRC_DIR}/Bindings/Fields.cpp"
        "${SRC_DIR}/Bindings/Operators.cpp"
        "${SRC_DIR}/Bindings/Poisson.cpp"
        "${SRC_DIR}/Bindings/Shapes.cpp"
        "${SRC_DIR}/Bindings/Simulation.cpp"
    )
    # Manually add pybind11 include dirs before those from Python.
    target_include_directories(${PYLIB} BEFORE PUBLIC ${pybind11_INCLUDE_DIRS})
    target_link_libraries(${PYLIB} PUBLIC ${CORE})
    target_link_libraries(${PYLIB} PRIVATE ${PRIVATEDEP})
endif()

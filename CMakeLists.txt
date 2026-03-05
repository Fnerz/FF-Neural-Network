cmake_minimum_required(VERSION 3.30.4)

project(main)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

set(CMAKE_BUILD_TYPE Debug)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rsc)

set(RSC_SOURCES
    rsc/ActivationFunctions.cpp
    rsc/ActivationLayer.cpp
    rsc/LayerDense.cpp
    rsc/LayerWrapper.cpp
    rsc/NeruralNetworkThread.cpp
    rsc/NeuralNetwork.cpp
    rsc/Optimizer.cpp
    rsc/OptimizerWrapper.cpp
    rsc/ThreadDevice.cpp
    rsc/WeightinitializationMethod.cpp
)

add_executable(main
    main.cpp
    ${RSC_SOURCES}
)

target_link_libraries(main)

set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

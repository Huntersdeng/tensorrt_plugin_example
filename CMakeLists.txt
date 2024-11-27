cmake_minimum_required(VERSION 3.5)
LIST(APPEND CMAKE_PROGRAM_PATH "/usr/local/cuda/bin/")
project(tensorrt_plugin_example LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release) # None, Debug, Release, RelWithDebInfo, MinSizeRel
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wno-unused-parameter -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS}  -Wno-unused-parameter -O3 -g -Wall")

find_package(CUDA REQUIRED)
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)

include_directories(${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} src)
add_library(identity_conv_plugin SHARED src/identity_conv_plugin.cpp 
                                        src/identity_conv_plugin_creator.cpp
                                        src/plugin_registration.cpp
                                        src/utils.cpp)
target_link_directories(identity_conv_plugin PUBLIC ${TensorRT_LIBRARIES})
target_link_libraries(identity_conv_plugin nvinfer nvinfer_plugin ${CUDA_LIBRARIES})

add_executable(build_engine build_engine.cpp)
target_link_libraries(build_engine PRIVATE nvinfer nvinfer_plugin nvonnxparser ${CUDA_LIBRARIES} identity_conv_plugin)

find_package(ament_cmake)
find_package(stardust_ai_engine REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)
# target_link_libraries(${PROJECT_NAME} identity_conv_plugin)
ament_target_dependencies(${PROJECT_NAME} stardust_ai_engine)

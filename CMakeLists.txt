cmake_minimum_required(VERSION 3.5)
LIST(APPEND CMAKE_PROGRAM_PATH "/usr/local/cuda/bin/")
project(tensorrt_plugin_example LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release) # None, Debug, Release, RelWithDebInfo, MinSizeRel
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wno-unused-parameter -Wno-deprecated-declarations -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS}  -Wno-unused-parameter -Wno-deprecated-declarations -O3 -g -Wall")

find_package(CUDA REQUIRED)
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)

include_directories(${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} src)

add_library(identity_conv_plugin_manual SHARED src/manual/identity_conv_plugin.cpp 
                                        src/manual/identity_conv_plugin_creator.cpp
                                        src/manual/plugin_registration.cpp
                                        src/manual/utils.cpp)
target_link_directories(identity_conv_plugin_manual PUBLIC ${TensorRT_LIBRARIES})
target_link_libraries(identity_conv_plugin_manual nvinfer nvinfer_plugin ${CUDA_LIBRARIES})

add_library(identity_conv_plugin_trtpg SHARED src/trtpg/IdentityconvIPluginV2DynamicExt/IdentityconvIPluginV2DynamicExt.cpp )
target_link_directories(identity_conv_plugin_trtpg PUBLIC ${TensorRT_LIBRARIES})
target_link_libraries(identity_conv_plugin_trtpg nvinfer nvinfer_plugin ${CUDA_LIBRARIES})

add_executable(build_engine build_engine.cpp)
target_link_libraries(build_engine PRIVATE nvinfer nvinfer_plugin nvonnxparser ${CUDA_LIBRARIES})

add_executable(${PROJECT_NAME} main.cpp src/trtpg/IdentityconvIPluginV2DynamicExt/IdentityconvIPluginV2DynamicExt.cpp)
target_link_libraries(${PROJECT_NAME} nvinfer nvinfer_plugin ${CUDA_LIBRARIES})

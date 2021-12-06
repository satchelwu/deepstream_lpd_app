cmake_minimum_required(VERSION 3.16.2)
project(nvdsinfer_platedetectioncustomparser)
file(GLOB SRC_FILES "*.h" "*.cpp")

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../models/plate_detection)  
include_directories("/opt/nvidia/deepstream/deepstream/sources/includes"
) 

include_directories(/usr/local/cuda/include)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
# add_custom_command(OUTPUT )
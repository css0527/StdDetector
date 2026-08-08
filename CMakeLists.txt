cmake_minimum_required(VERSION 3.16)
project(StdDetector)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# 全局包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    include
    detector
    predict
    tools
    io
)

# 添加io子目录
add_subdirectory(io)

# 源文件
set(SOURCES
    main.cpp
    detector/detector.cpp
    detector/light_corner_corrector.cpp
    detector/pnp_optimizer.cpp
    predict/kalman_filter.cpp
    tools/img_tools.cpp
    tools/logger.cpp
    tools/math_tools.cpp
    tools/crc.cpp
)

# 创建可执行文件
add_executable(main ${SOURCES})

# 链接库
target_link_libraries(main 
    io
    ${OpenCV_LIBS}
    fmt::fmt
    yaml-cpp
    ${Eigen3_LIBS}
)

# 编译选项
target_compile_options(main PRIVATE -Wall -Wextra -O2)

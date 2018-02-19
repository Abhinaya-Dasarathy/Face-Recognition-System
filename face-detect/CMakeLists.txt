cmake_minimum_required(VERSION 2.8)
project(Face)
set(PROJECT_SRC  OpenCV_Face_Detect.cpp FaceDetectMain.cu GPU_Face_Detect_Kernels.cu GPU_Face_Detect.cu)
set(PROJECT_HDR  OpenCV_Face_Detect.h FaceDetectExtras.h GPU_Face_Detect.cuh GPUHaarCascade.h lock.h)

find_package( OpenCV REQUIRED)
find_package( CUDA REQUIRED)
include_directories(/data/apps/cuda/6.0/include)

cuda_add_Executable( Face ${PROJECT_SRC} ${PROJECT_HDR})
target_link_libraries( Face ${OpenCV_LIBS} -L./libs -L/data/apps/cuda/6.0/lib64 -lpthread ${CUDA_nvcuvid_LIBRARY} ${CUDA_DRIVER_LIBRARY_PATH})

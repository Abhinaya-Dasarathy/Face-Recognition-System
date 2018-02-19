cmake_minimum_required(VERSION 2.8)
project(ParallelFaceRecog)

# set source files

set(PROJECT_SRC  mpi_eigenfaces.cpp)


find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)
find_package( CUDA REQUIRED)
include_directories(/data/apps/cuda/6.0/include)
include_directories(/data/apps/mpi/openmpi-1.8.3/gcc/4.9.2/include)




cuda_add_executable(ParallelFaceRecog ${PROJECT_SRC} )
target_link_libraries(ParallelFaceRecog  ${OpenCV_LIBS} -L./libs -L/data/apps/cuda/6.0/lib64 -lpthread ${CUDA_nvcuvid_LIBRARY} ${CUDA_DRIVER_LIBRARY_PATH} ${MPI_LIBRARIES} )

if(MPI_COMPILE_FLAGS)
  set_target_properties(ParallelFaceRecog PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(ParallelFaceRecog  PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
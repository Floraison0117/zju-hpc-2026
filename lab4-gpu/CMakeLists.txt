cmake_minimum_required(VERSION 3.20)
project(amss_ncku_lab LANGUAGES CXX Fortran)

# --- Options ----------------------------------------------------------------
option(AMSS_ENABLE_GPU "Enable CUDA GPU build (ABEGPU target)" OFF)
set(AMSS_OPT "-O3" CACHE STRING "C/C++/Fortran optimization level")
set(AMSS_ARCH_FLAGS "" CACHE STRING "Architecture-specific compiler flags")
option(AMSS_ENABLE_OPENMP "Link OpenMP to all executables" OFF)
set(AMSS_MPI_CUDA_AWARE 0 CACHE STRING "Set MPI_CUDA_AWARE macro for ABEGPU (0/1)")
set_property(CACHE AMSS_MPI_CUDA_AWARE PROPERTY STRINGS 0 1)

# --- Compile flags ----------------------------------------------------------
# Legacy AMSS-NCKU has strict-aliasing UB; -fno-strict-aliasing is kept
# regardless of AMSS_OPT so changing -O does not lose the safety flag.
# `-Dfortran3 -Dnewc` select the modern C/Fortran source branches (without
# them the C++ side falls back to pre-ANSI <iostream.h> includes and the
# Fortran side skips the f90 modules). Apply to every supported compiler.
separate_arguments(AMSS_OPT_LIST UNIX_COMMAND "${AMSS_OPT}")
separate_arguments(AMSS_ARCH_FLAGS_LIST UNIX_COMMAND "${AMSS_ARCH_FLAGS}")

# Clang/IntelLLVM/ARMClang all accept the GCC-style flags used here.
# COMPILE_LANG_AND_ID takes comma-separated compiler IDs (not CMake list).
add_compile_options(
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,IntelLLVM,Intel,ARMClang>:${AMSS_OPT_LIST};${AMSS_ARCH_FLAGS_LIST};-fno-strict-aliasing;-Wno-deprecated;-Dfortran3;-Dnewc>"
  "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,Flang,IntelLLVM,Intel,ARMClang>:${AMSS_OPT_LIST};${AMSS_ARCH_FLAGS_LIST};-fno-strict-aliasing;-cpp>"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- MPI --------------------------------------------------------------------
find_package(MPI REQUIRED COMPONENTS CXX)

# --- Shared Fortran sources -------------------------------------------------
set(AMSS_F90SOURCES
  src/bssn_rhs.f90
  src/diff_new.f90 src/enforce_algebra.f90
  src/fadmquantites_bssn.f90 src/fmisc.f90
  src/getnp4.f90
  src/initial_puncture.f90 src/kodiss.f90
  src/lopsidediff.f90
  src/prolongrestrict_cell.f90 src/rungekutta4_rout.f90
  src/sommerfeld_rout.f90
)

# --- ABE: CPU main evolution executable -------------------------------------
set(ABE_SOURCES
  src/ABE.C src/Ansorg.C src/Block.C src/MPatch.C src/Parallel.C
  src/bssn_class.C src/cgh.C src/checkpoint.C
  src/misc.C src/monitor.C src/perf.C
  src/surface_integral.C src/transpbh.C src/var.C
  ${AMSS_F90SOURCES}
)
add_executable(ABE ${ABE_SOURCES})
target_link_libraries(ABE PRIVATE MPI::MPI_CXX)

# --- TwoPunctureABE: initial-data solver -----------------------------------
add_executable(TwoPunctureABE src/TwoPunctureABE.C src/TwoPunctures.C)
target_link_libraries(TwoPunctureABE PRIVATE MPI::MPI_CXX)
option(AMSS_ENABLE_TWOP_COS_TABLE "Cache Chebyshev cosine tables in TwoPuncture (bit-exact)" ON)
if(AMSS_ENABLE_TWOP_COS_TABLE)
  target_compile_definitions(TwoPunctureABE PRIVATE AMSS_ENABLE_TWOP_COS_TABLE)
endif()

# --- Optional OpenMP --------------------------------------------------------
if(AMSS_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX Fortran)
  target_link_libraries(ABE PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_Fortran)
  target_link_libraries(TwoPunctureABE PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- TwoPuncture GPU acceleration (bonus) -----------------------------------
# TwoPunctureABE GPU path is decoupled from ABEGPU: AMSS_ENABLE_TWOP_GPU (default ON)
# controls whether TwoPunctureABE uses the device kernels. When OFF, TwoPunctureABE
# builds CPU-only and can use OpenMP (AMSS_ENABLE_TWOP_OMP_TUNE + AMSS_ENABLE_PACKED_RELAX).
option(AMSS_ENABLE_TWOP_GPU "Use GPU for TwoPuncture J_times_dv/F_of_v (on by default for backward compat)" ON)
option(AMSS_ENABLE_TWOP_OMP_TUNE "Hoist a single OpenMP team for TwoPuncture relax() (bit-exact, CPU path)" OFF)
option(AMSS_ENABLE_PACKED_RELAX "Use packed JFD views in TwoPuncture relaxation (bit-exact, CPU path)" ON)
if(AMSS_ENABLE_TWOP_GPU)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  target_sources(TwoPunctureABE PRIVATE src/twopunctures_gpu.cu)
  set_source_files_properties(src/twopunctures_gpu.cu PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(TwoPunctureABE PRIVATE USE_GPU)
  target_link_libraries(TwoPunctureABE PRIVATE CUDA::cudart)
  option(AMSS_ENABLE_TWOP_GPU_ASYNC "Async CUDA stream + coalesced pinned transfers for TwoPuncture GPU" ON)
  if(AMSS_ENABLE_TWOP_GPU_ASYNC)
    target_compile_definitions(TwoPunctureABE PRIVATE USE_GPU_ASYNC)
  endif()
endif()
# CPU path: OpenMP + optional OMP_TUNE/packed_relax (only effective when TWOP_GPU=OFF)
target_compile_options(TwoPunctureABE PRIVATE -fopenmp)
target_link_options(TwoPunctureABE PRIVATE -fopenmp)
if(AMSS_ENABLE_PACKED_RELAX)
  target_compile_definitions(TwoPunctureABE PRIVATE AMSS_ENABLE_PACKED_RELAX)
endif()
if(AMSS_ENABLE_TWOP_OMP_TUNE)
  target_compile_definitions(TwoPunctureABE PRIVATE AMSS_ENABLE_TWOP_OMP_TUNE)
endif()

# --- ABEGPU: GPU evolution executable (device-resident baseline) -----------
# Same C++/Fortran sources as ABE except bssn_class.C → bssn_gpu_class.C,
# plus GPU kernel/agent files.
set(ABEGPU_CXX_SOURCES
  src/ABE.C src/Ansorg.C src/Block.C src/MPatch.C src/Parallel.C
  src/bssn_gpu_class.C src/bssn_step_gpu.C
  src/cgh.C src/checkpoint.C src/misc.C
  src/monitor.C src/perf.C src/surface_integral.C
  src/transpbh.C src/var.C
  src/helper.cpp src/Parallel_GPU.cpp
)
set(ABEGPU_CUDA_SOURCES
  src/fmisc_gpu.cu src/lopsidediff_gpu.cu src/kodiss_gpu.cu src/diff_new_gpu.cu
  src/bssn_rhs_gpu.cu src/bssn_rhs_gpu_int.cu src/bssn_rhs_gpu_face.cu src/bssn_rhs_gpu_facez.cu src/prolongrestrict_cell_gpu.cu src/prolongrestrict_cell_gpu_int.cu src/gpu_manager.cu
  src/enforce_algebra_gpu.cu src/sommerfeld_rout_gpu.cu src/rungekutta4_rout_gpu.cu
  src/fadmquantites_bssn_gpu.cu src/misc_gpu.cu src/MPatch_gpu.cu
  src/surface_integral_gpu.cu src/getnp4_gpu.cu
)

if(AMSS_ENABLE_GPU)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  find_package(CUDAToolkit REQUIRED)

  add_executable(ABEGPU ${ABEGPU_CXX_SOURCES} ${ABEGPU_CUDA_SOURCES} ${AMSS_F90SOURCES})
  set_source_files_properties(${ABEGPU_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_target_properties(ABEGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(ABEGPU PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true;-lineinfo>")
# Cap stencil file registers at 128 to avoid nvlink overflow (d_fdderivs_point hits 132 with newer nvcc)
set_source_files_properties(src/diff_new_gpu.cu src/lopsidediff_gpu.cu src/kodiss_gpu.cu PROPERTIES
  COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=128>")
  target_compile_definitions(ABEGPU PRIVATE USE_GPU fortran3 newc MPI_CUDA_AWARE=${AMSS_MPI_CUDA_AWARE})
  target_include_directories(ABEGPU PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# MPI includes must be explicit for CUDA language files (nvcc does not inherit MPI::MPI_CXX imported target includes)
target_include_directories(ABEGPU PRIVATE ${MPI_CXX_INCLUDE_DIRS})
# fallback for environments where MPI_CXX_INCLUDE_DIRS is not populated
find_path(MPI_INC_OPENMPI mpi.h PATHS /usr/lib/x86_64-linux-gnu/openmpi/include /usr/include NO_DEFAULT_PATH)
if(MPI_INC_OPENMPI)
  target_include_directories(ABEGPU PRIVATE ${MPI_INC_OPENMPI} /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi)
endif()
  target_link_libraries(ABEGPU PRIVATE CUDA::cudart MPI::MPI_CXX)
  if(AMSS_ENABLE_OPENMP)
    target_link_libraries(ABEGPU PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_Fortran)
  endif()
endif()

# --- Configure summary ------------------------------------------------------
message(STATUS "==================== AMSS-NCKU Lab 4 ====================")
message(STATUS "CXX      : ${CMAKE_CXX_COMPILER}")
message(STATUS "Fortran  : ${CMAKE_Fortran_COMPILER}")
if(AMSS_ENABLE_GPU)
  message(STATUS "GPU      : ON, CUDA=${CMAKE_CUDA_COMPILER}, arch=${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "GPU      : OFF")
endif()
message(STATUS "AMSS_OPT=${AMSS_OPT} ARCH_FLAGS=${AMSS_ARCH_FLAGS} OPENMP=${AMSS_ENABLE_OPENMP} MPI_CUDA_AWARE=${AMSS_MPI_CUDA_AWARE}")
message(STATUS "==========================================================")

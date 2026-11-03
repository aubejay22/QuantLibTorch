cmake_minimum_required(VERSION 3.24)
project(QuantLibTorch VERSION 0.3.0 LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(QLT_QUANTLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/QuantLib" CACHE PATH "Pinned QuantLib sources")
set(QLT_BOOST_INCLUDE_DIR "" CACHE PATH "Directory containing boost/version.hpp")
set(QLT_COMPILE_JOBS "2" CACHE STRING "Maximum parallel compiler processes on MSVC")
if(NOT QLT_COMPILE_JOBS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "QLT_COMPILE_JOBS must be a positive integer")
endif()
option(QLT_PATCH_MATH "Enable minimal ADL math compatibility overlay" ON)
if(NOT EXISTS "${QLT_QUANTLIB_DIR}/ql/version.hpp")
    message(FATAL_ERROR "QuantLib sources missing. Run python scripts/bootstrap.py")
endif()

# Use the libraries shipped with the active Python's torch. This CPU-only
# extension does not need a CUDA toolkit, even when torch is a CUDA wheel.
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import torch, sys; print(torch.__path__[0]); print(int(torch._C._GLIBCXX_USE_CXX11_ABI)); print(sys.base_prefix)"
    OUTPUT_VARIABLE TORCH_INFO OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" TORCH_INFO "${TORCH_INFO}")
list(GET TORCH_INFO 0 TORCH_ROOT)
list(GET TORCH_INFO 1 TORCH_ABI)
list(GET TORCH_INFO 2 PYTHON_BASE)
file(TO_CMAKE_PATH "${TORCH_ROOT}" TORCH_ROOT)
file(TO_CMAKE_PATH "${PYTHON_BASE}" PYTHON_BASE)

if(NOT QLT_BOOST_INCLUDE_DIR)
    get_filename_component(PYTHON_DIR "${Python3_EXECUTABLE}" DIRECTORY)
    find_path(BOOST_HEADERS boost/version.hpp HINTS "${PYTHON_DIR}/Library/include" "${PYTHON_BASE}/Library/include" ENV BOOST_ROOT)
    if(NOT BOOST_HEADERS)
        message(FATAL_ERROR "Boost headers missing; set QLT_BOOST_INCLUDE_DIR")
    endif()
    set(QLT_BOOST_INCLUDE_DIR "${BOOST_HEADERS}")
endif()

include(cmake/quantlib_sources.cmake)
add_library(quantlibtorch_native SHARED cpp/build_info.cpp cpp/bridge.cpp cpp/fd_bridge.cpp cpp/adjoint.cpp
    cpp/scalar_checks.cpp cpp/tensor_kernels.cpp cpp/kernel_checks.cpp cpp/spline_kernels.cpp cpp/spline_checks.cpp
    cpp/parallel_solve.cpp cpp/step_adjoint.cpp cpp/pcr_stage.cpp cpp/solve_checks.cpp cpp/operator_arithmetic.cpp
    cpp/thomas_arithmetic.cpp cpp/step_arithmetic.cpp cpp/time_variance.cpp cpp/step_preparation.cpp
    cpp/structured_ops.cpp cpp/pde_probe.cpp cpp/coefficient_pde.cpp cpp/packed_pde.cpp)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/native_metadata.py"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND_ERROR_IS_FATAL ANY)
target_include_directories(quantlibtorch_native PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_packed.py"
    "${QLT_QUANTLIB_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/packed-quantlib"
    COMMAND_ERROR_IS_FATAL ANY)
target_include_directories(quantlibtorch_native PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/packed-quantlib")
target_compile_features(quantlibtorch_native PRIVATE cxx_std_17)
target_compile_definitions(quantlibtorch_native PRIVATE
    QL_INCLUDE_FIRST=quantlibtorch/torch_real.hpp QL_NULL_AS_FUNCTIONS QL_NO_AUTO_LINK NOMINMAX)
target_precompile_headers(quantlibtorch_native PRIVATE <ql/types.hpp>)
# Predicates and elementary operator forwards preserve separately rounded
# upstream operations. Keep this policy local, outside the shared PCH.
set_source_files_properties(cpp/solve_checks.cpp cpp/operator_arithmetic.cpp cpp/thomas_arithmetic.cpp cpp/step_arithmetic.cpp cpp/time_variance.cpp cpp/step_preparation.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
if(MSVC)
    set_source_files_properties(cpp/solve_checks.cpp cpp/operator_arithmetic.cpp cpp/thomas_arithmetic.cpp cpp/step_arithmetic.cpp cpp/time_variance.cpp cpp/step_preparation.cpp PROPERTIES COMPILE_OPTIONS /fp:strict)
else()
    set_source_files_properties(cpp/solve_checks.cpp cpp/operator_arithmetic.cpp cpp/thomas_arithmetic.cpp cpp/step_arithmetic.cpp cpp/time_variance.cpp cpp/step_preparation.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()
if(QLT_PATCH_MATH)
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_quantlib.py"
        "${QLT_QUANTLIB_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/quantlib-overlay"
        COMMAND_ERROR_IS_FATAL ANY)
    target_include_directories(quantlibtorch_native BEFORE PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/quantlib-overlay")
endif()

foreach(SOURCE IN LISTS QLT_QUANTLIB_SOURCES)
    if(QLT_PATCH_MATH AND EXISTS "${CMAKE_CURRENT_BINARY_DIR}/quantlib-overlay/${SOURCE}")
        target_sources(quantlibtorch_native PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/quantlib-overlay/${SOURCE}")
    else()
        target_sources(quantlibtorch_native PRIVATE "${QLT_QUANTLIB_DIR}/${SOURCE}")
    endif()
endforeach()
target_include_directories(quantlibtorch_native PRIVATE cpp/include "${QLT_QUANTLIB_DIR}")
target_include_directories(quantlibtorch_native SYSTEM PRIVATE
    "${QLT_BOOST_INCLUDE_DIR}" "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")
foreach(LIBRARY torch torch_cpu c10)
    # The active Python can change between configurations; never reuse an old
    # torch library path with headers from a new environment.
    unset(TORCH_${LIBRARY}_LIBRARY CACHE)
    find_library(TORCH_${LIBRARY}_LIBRARY NAMES ${LIBRARY} PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH NO_CACHE REQUIRED)
    target_link_libraries(quantlibtorch_native PRIVATE "${TORCH_${LIBRARY}_LIBRARY}")
endforeach()
if(MSVC)
    target_compile_options(quantlibtorch_native PRIVATE /EHsc /bigobj /utf-8 /Zc:__cplusplus /MP${QLT_COMPILE_JOBS})
    # __FILE__ in upstream diagnostics must not disclose developer home paths.
    # MSVC requires deterministic mode before it honors /pathmap.
    target_compile_options(quantlibtorch_native PRIVATE /experimental:deterministic
        "/pathmap:${CMAKE_CURRENT_SOURCE_DIR}=QuantLibTorch"
        "/pathmap:${TORCH_ROOT}=torch"
        "/pathmap:${QLT_BOOST_INCLUDE_DIR}=boost-headers")
    set_property(TARGET quantlibtorch_native PROPERTY MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
else()
    target_compile_definitions(quantlibtorch_native PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})
    target_compile_options(quantlibtorch_native PRIVATE -fvisibility=hidden
        "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=QuantLibTorch"
        "-ffile-prefix-map=${TORCH_ROOT}=torch"
        "-ffile-prefix-map=${QLT_BOOST_INCLUDE_DIR}=boost-headers")
    # The installed torch dependency supplies LibTorch. Never embed a developer
    # machine's absolute environment path into the distributed shared library.
    set_property(TARGET quantlibtorch_native PROPERTY BUILD_RPATH_USE_ORIGIN TRUE)
    set_property(TARGET quantlibtorch_native PROPERTY BUILD_WITH_INSTALL_RPATH TRUE)
    set_property(TARGET quantlibtorch_native PROPERTY INSTALL_RPATH "$ORIGIN/../../torch/lib")
endif()
set_target_properties(quantlibtorch_native PROPERTIES PREFIX "" OUTPUT_NAME "quantlibtorch_native"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/quantlibtorch/_native"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/quantlibtorch/_native")
foreach(CONFIG Release Debug RelWithDebInfo MinSizeRel)
    string(TOUPPER "${CONFIG}" CONFIG_UPPER)
    set_target_properties(quantlibtorch_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}/python/quantlibtorch/_native"
        RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}/python/quantlibtorch/_native")
endforeach()
enable_testing()
add_test(NAME python-tests COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set_tests_properties(python-tests PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

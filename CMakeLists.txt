# CMakeLists 
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
cmake_policy(SET CMP0079 NEW) # Allow linking with targets in other directories
project(
    micromesh_toolkit
    VERSION 1.1
    DESCRIPTION "Micromesh Tools"
    LANGUAGES C CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/_install" CACHE PATH "folder in which INSTALL will put everything needed to run the binaries" FORCE)

# Set the default for enabling beta extensions before nvpro_core. Beta
# extensions can conflict if the Vulkan SDK and nvpro_core generated
# nvvk/extensions_vk.hpp versions are different.
option(VK_ENABLE_BETA_EXTENSIONS "Enable beta extensions provided by the Vulkan SDK" OFF)

# Enable bounds and other error checking for gcc debug builds in std::vector
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_GLIBCXX_DEBUG>)

# Look for nvpro_core 1) as a sub-folder 2) at some other locations
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external ${CMAKE_CURRENT_SOURCE_DIR}/..
    REQUIRED
    DOC "Directory containing nvpro_core"
  )
endif()

if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  set(OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin_x64)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

set(NVPRO_CORE_DIR ${BASE_DIRECTORY}/nvpro_core)

if(MSVC)
    add_definitions(/wd26812)  # 'enum class' over 'enum'
    add_definitions(/wd26451)  # Arithmetic overflow, casting 4 byte value to 8 byte value
else()
  # Ignore warnings from some third party libraries by including them with -isystem
  include_directories(SYSTEM
    external/nvpro_core/
    external/nvpro_core/imgui/
    external/nvpro_core/third_party/stb/
    external/nvpro_core/third_party/imgui/
    external/nvpro_core/third_party/vma/include/
    external/nvpro_core/third_party/tinygltf/
    micromesh_toolbox/thirdparty/aftermath_sdk/include/
    )
endif()

#--------------------------------------------------------------------------------------------------
# Package shared by all projects

# set(SUPPORT_AFTERMATH on)
# _add_package_NsightAftermath()
_add_package_ZLIB()
_add_package_VulkanSDK()
_add_package_ShaderC()
_add_package_ImGUI()
if(WIN32)
	_add_package_DirectX12()
endif()
_add_nvpro_core_lib()

_add_package_Glm()
# nvpro_core hides GLM_INCLUDE_DIR, but it's commonly missing if not included with the vulkan SDK
mark_as_advanced(CLEAR GLM_INCLUDE_DIR)
if(NOT GLM_FOUND)
  message(FATAL_ERROR "GLM not found. Aborting. Set `GLM_INCLUDE_DIR` or include `GLM` with the Vulkan SDK install")
endif()
  
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

# Override BARY_CORE_BUILD_UTILS. bary_utils is needed by meshops_core
set(BARY_CORE_BUILD_UTILS "ON" CACHE INTERNAL "meshops_core requires bary_utils")

function(find_first RESULT NAME)
  foreach(DIR ${ARGN})
    if(EXISTS ${DIR})
      message(STATUS "Dependency ${NAME} found at ${DIR}")
      set(${RESULT} ${DIR} PARENT_SCOPE)
      return()
    else()
      message(STATUS "Dependency ${NAME} not found at ${DIR}")
    endif()
  endforeach()
  message(FATAL_ERROR "${NAME} not found")
endfunction()

find_first(PATH_BARY_CORE Displacement-MicroMap-BaryFile
  ${CMAKE_CURRENT_SOURCE_DIR}/external/Displacement-MicroMap-BaryFile
  ${CMAKE_CURRENT_SOURCE_DIR}/../bary_core
  ${CMAKE_CURRENT_SOURCE_DIR}/../sdk/bary_core
  )
add_subdirectory(${PATH_BARY_CORE} ${CMAKE_BINARY_DIR}/bary_core)

find_first(PATH_MICROMESH_CORE Displacement-MicroMap-SDK/micromesh_core
  ${CMAKE_CURRENT_SOURCE_DIR}/external/Displacement-MicroMap-SDK/micromesh_core
  ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_core
  ${CMAKE_CURRENT_SOURCE_DIR}/../sdk/micromesh_core
  )
add_subdirectory(${PATH_MICROMESH_CORE} ${CMAKE_BINARY_DIR}/micromesh_core)

find_first(PATH_SDK_COMPRESSION Displacement-MicroMap-SDK/micromesh_displacement_compression
  ${CMAKE_CURRENT_SOURCE_DIR}/external/Displacement-MicroMap-SDK/micromesh_displacement_compression
  ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_displacement_compression
  ${CMAKE_CURRENT_SOURCE_DIR}/../sdk/micromesh_displacement_compression
  )
add_subdirectory(${PATH_SDK_COMPRESSION} ${CMAKE_BINARY_DIR}/micromesh_displacement_compression)

find_first(PATH_SDK_REMESHING Displacement-MicroMap-SDK/micromesh_displacement_remeshing
  ${CMAKE_CURRENT_SOURCE_DIR}/external/Displacement-MicroMap-SDK/micromesh_displacement_remeshing
  ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_displacement_remeshing
  ${CMAKE_CURRENT_SOURCE_DIR}/../sdk/micromesh_displacement_remeshing
  )
add_subdirectory(${PATH_SDK_REMESHING} ${CMAKE_BINARY_DIR}/micromesh_displacement_remeshing)

###
# Custom build modifications for libpng.
option(PNG_BUILD_ZLIB "Custom zlib Location, else find_package is used" ON)
option(PNG_EXECUTABLES "Build libpng executables" OFF)
option(PNG_SHARED "Build shared PNG lib" OFF)
option(PNG_TESTS "Build libpng tests" OFF)
set(SKIP_INSTALL_ALL ON) # Skip all libpng install targets
# This is a bit of a hack: we must include zlibstatic from nvpro_core, so we
# must also turn off libpng's build configuration (since this configuration
# fails on WSL, and since it produces a compile-time error if the detected zlib
# version doesn't match nvpro_core's zlib version). To do this, we set IOS
# around libpng's CMake files.
set(IOS ON)
add_subdirectory(external/libpng)
unset(IOS)
target_link_libraries(png_static zlibstatic)
target_include_directories(png_static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/external/libpng> ${CMAKE_CURRENT_LIST_DIR}/external/libpng)
###
add_subdirectory(external/tinyexr)
add_subdirectory(meshops_core)
add_subdirectory(meshops_bake)
add_subdirectory(meshops_remesher)
add_subdirectory(mini_samples)
add_subdirectory(micromesh_tool)
add_subdirectory(micromesh_python)
add_subdirectory(micromesh_toolbox)

# Enable some additional warnings for micromesh toolkit projects Attempt to
# match warnings between different compilers
if(WIN32) # Windows
  set(MICROMESH_COMPILE_WARNINGS
    )
else() # Linux
  set(MICROMESH_COMPILE_WARNINGS
    -Werror
    -Werror=sign-compare
    -Werror=conversion
    -Werror=format
    -Werror=unused
    )
endif()

# Add per-source file compile options because many third party dependencies
# are not compatible with more strict warnings.
foreach(TARGET
  meshops_core
  meshops_bake
  meshops_remesher
  dmm_displacement
  micromesh_tool_lib
  micromesh_tool
  micromesh_python
  micromesh_toolbox
  )
  if (TARGET ${TARGET})
    set_property(TARGET ${TARGET} APPEND PROPERTY COMPILE_OPTIONS ${MICROMESH_COMPILE_WARNINGS})
  endif()
endforeach()

set_target_properties(bary_core PROPERTIES FOLDER "bary_libs")
set_target_properties(bary_utils PROPERTIES FOLDER "bary_libs")

set_target_properties(meshops_core PROPERTIES FOLDER "meshops_libs")
set_target_properties(meshops_bake PROPERTIES FOLDER "meshops_libs")
set_target_properties(meshops_remesher PROPERTIES FOLDER "meshops_libs")

set_target_properties(micromesh_core PROPERTIES FOLDER "micromesh_libs")
set_target_properties(micromesh_displacement_compression PROPERTIES FOLDER "micromesh_libs")
set_target_properties(micromesh_displacement_remeshing PROPERTIES FOLDER "micromesh_libs")

set_target_properties(micromesh_tool_lib PROPERTIES FOLDER "tool_libs")
if (TARGET micromesh_python)
  set_target_properties(micromesh_python PROPERTIES FOLDER "tool_libs")
endif()

set_target_properties(micromesh_tool PROPERTIES FOLDER "tools")
set_target_properties(micromesh_toolbox PROPERTIES FOLDER "tools")

set_target_properties(nvpro_core PROPERTIES FOLDER "ThirdParty")
set_target_properties(tinyexr PROPERTIES FOLDER "ThirdParty")
set_target_properties(png_static PROPERTIES FOLDER "ThirdParty")
set_target_properties(genfiles PROPERTIES FOLDER "ThirdParty") # From libpng

install(DIRECTORY "media" DESTINATION ${CMAKE_INSTALL_BINDIR})

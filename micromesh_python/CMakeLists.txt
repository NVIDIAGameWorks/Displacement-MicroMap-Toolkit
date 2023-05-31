cmake_minimum_required(VERSION 3.5)

if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../..
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

#_add_project_definitions(${PROJNAME})
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/packman.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/packman.cmake)
endif()

if(USE_PACKMAN)
  # Use custom all-deps file instead of nvpro_core all-deps to target appropriate nv-usd builds
  configure_all_deps_file(PYTHON_VERSION_STRING "py310" ALL_DEPS_FILEPATH "${CMAKE_CURRENT_SOURCE_DIR}/all-deps.packman.xml")

  find_package(USD)

  # Ensure expected python is available
  pull_dependencies(DEPENDENCY_FILE "python-deps.packman.xml")

  # Stop looking fore newer versions in other locations
  set(Python3_FIND_STRATEGY LOCATION)
  # On Windows, ignore a potentially preinstalled newer version of Python
  set(Python3_FIND_REGISTRY NEVER)

  set(Python3_ROOT_DIR "${BASE_DIRECTORY}/nvpro_core/OV/downloaded/python")

  #message(STATUS "Python3_ROOT_DIR ${Python3_ROOT_DIR}")
  find_package(Python3 COMPONENTS Development Interpreter)
else()
  # Only py37 supported here because public nv-usd package is built for python 3.7
  set(PYTHON_VERSION "py37" CACHE INTERNAL "")
  find_package(USD)
  find_package(Python3 COMPONENTS Development Interpreter)
endif()

#message(STATUS "Python3_EXECUTABLE ${Python3_EXECUTABLE}")
#message(STATUS "Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS}")
#message(STATUS "Python3_RUNTIME_LIBRARY_DIRS ${Python3_RUNTIME_LIBRARY_DIRS}")
#message(STATUS "Python3_LIBRARY_DIRS ${Python3_LIBRARY_DIRS}")
#message(STATUS "Python3_LIBRARIES ${Python3_LIBRARIES}")

# If Python3 is not found, this project will be excluded from build

# If Pybind11 is not found, this project will be excluded from build
find_package(Pybind11)

if(USE_PACKMAN)
  # Install packman-packager
  message(STATUS "\"${BASE_DIRECTORY}/nvpro_core/OV/packman/${PACKMAN_COMMAND}\" install packman-packager@0.10 -l ${CMAKE_CURRENT_BINARY_DIR}/packman-packager")
  execute_process(COMMAND "${BASE_DIRECTORY}/nvpro_core/OV/packman/${PACKMAN_COMMAND}" install packman-packager@0.10 -l ${CMAKE_CURRENT_BINARY_DIR}/packman-packager
                  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                  RESULT_VARIABLE PACKMAN_RESULT)
  if (${PACKMAN_RESULT} EQUAL 0)
    message(STATUS "Packman result: success")
  else()
    message(FATAL_ERROR "Packman result: ${PACKMAN_RESULT}")
  endif()          
  set(MICROMESH_PYTHON_VERSION 100.2211.1 CACHE STRING "100.2211.1")  # <bary version.usd version.minor version>
endif()

set(PROJECT_NAME "micromesh_python")

# Only build micromesh_python if its dependencies are found and it's not explicitly disabled
if((Python3_FOUND AND EXISTS ${PYBIND11_INCLUDE_DIR}) AND ((NOT DEFINED BUILD_MICROMESH_PYTHON) OR BUILD_MICROMESH_PYTHON))

  # Set micromesh_python to be included in the build (default on)
  option(BUILD_MICROMESH_PYTHON "Require micromesh_python" ON)

  #-------------------------------------------------------------------------
  # Library
  #-------------------------------------------------------------------------

  # Sources and headers for the library
  file(GLOB SOURCE_FILES *.cpp)
  file(GLOB HEADER_FILES *.h *.hpp)

  # Attempt to match MSVC warnings
  if(UNIX)
      set_property(SOURCE ${SOURCE_FILES} APPEND_STRING PROPERTY COMPILE_FLAGS -Wconversion)
  endif()

  _add_package_VMA()
  #_add_package_VulkanSDK()

  # Add library
  add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})

  # Udjust python lib name and suffix (if required)
  set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
  if (WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
  endif()

  # Add thirdparty  
  target_include_directories(${PROJECT_NAME} PRIVATE ${THIRDPARTY_DIR})
  target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_DIR})
  target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIR})
  target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
  target_include_directories(${PROJECT_NAME} PRIVATE ${BASE_DIRECTORY}/nvpro_core/third_party/stb)
  target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_tool/src)
  target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_core/include)
  #target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../micromesh_tool/in)

  target_link_directories(${PROJECT_NAME} PUBLIC "${Python3_LIBRARY_DIRS}")

  # Dependencies
  target_link_libraries (${PROJECT_NAME} PUBLIC meshops_bake meshops_core micromesh_core micromesh_tool_lib bary_core bary_utils "${Python3_LIBRARIES}")

  if (WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE /bigobj)
  else()
    target_compile_options(${PROJECT_NAME} PRIVATE -fvisibility=hidden)
  endif()

  # Adding extra source files from nvpro_core
  target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_NAME="${PROJECT_NAME}")
  target_sources(${PROJECT_NAME} PRIVATE ${COMMON_SOURCE_FILES})
  source_group("nvpro_core"     FILES ${COMMON_SOURCE_FILES})

  install(TARGETS ${PROJECT_NAME})

  # Utility scripts
  if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(USD_CONFIG_LIB_DIR "${USD_DEBUG_LIB_DIR}")
  else()
    set(USD_CONFIG_LIB_DIR "${USD_RELEASE_LIB_DIR}")
  endif()

  message("USD_CONFIG_LIB_DIR ${USD_CONFIG_LIB_DIR}")

  if (WIN32)
    set(SET_ENV_CMD "set")
    set(LIBRARY_PATH_SETTING "PATH=%PATH%")
    if (EXISTS ${USD_CONFIG_LIB_DIR})
      set(PYTHON_PATH ;${USD_CONFIG_LIB_DIR}\\python;${CMAKE_CURRENT_BINARY_DIR})
      set(LIBRARY_PATH_SETTING "PATH=${USD_CONFIG_LIB_DIR};%PATH%")
    endif()
    set(PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}/Release;${PYTHON_PATH})
    set(PYTHON_EXE "\"${Python3_EXECUTABLE}\"")
    set(PYTHON_EXE_ARGS "\"${Python3_EXECUTABLE}\" %*")
  else()
    set(SET_ENV_CMD "export")
    set(PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}:$PYTHONPATH")
    set(LIBRARY_PATH_SETTING "LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
    if (EXISTS ${USD_CONFIG_LIB_DIR})
      set(PYTHON_PATH "${USD_CONFIG_LIB_DIR}/python:${CMAKE_CURRENT_BINARY_DIR}/nvMicromesh:${PYTHON_PATH}")
      set(LIBRARY_PATH_SETTING "LD_LIBRARY_PATH=${USD_CONFIG_LIB_DIR}:$LD_LIBRARY_PATH")
    endif()
    set(PYTHON_EXE "${Python3_EXECUTABLE}")
    set(PYTHON_EXE_ARGS "\"${Python3_EXECUTABLE}\" $@")
  endif()
  
  configure_file("install_jupyterlab.cmd.in" "${CMAKE_CURRENT_BINARY_DIR}/install_jupyterlab.cmd" @ONLY)
  configure_file("start_jupyterlab.cmd.in" "${CMAKE_CURRENT_BINARY_DIR}/start_jupyterlab.cmd" @ONLY)
  configure_file("python.cmd.in" "${CMAKE_CURRENT_BINARY_DIR}/python.cmd" @ONLY)
  

  # package target to create and publish a packman package (append -f to command below to force publish)
  if (WIN32)
    set(BUILD_PLATFORM "windows-x86_64")
    set(MODULE_EXT "pyd")
  else()
    set(BUILD_PLATFORM "linux-x86_64")
    set(MODULE_EXT "so")
  endif()

  string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/package.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/package.py
    @ONLY
  )

  if (USE_PACKMAN)
    set(PACKAGE_TARGET micromesh_python_package)
    add_custom_target(${PACKAGE_TARGET})
    set_target_properties(${PACKAGE_TARGET} PROPERTIES FOLDER packman-packages)
    add_custom_command(TARGET ${PACKAGE_TARGET}
      COMMAND "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_BINARY_DIR}/package.py -b ${CMAKE_CURRENT_BINARY_DIR}/package -r cloudfront
    )
  endif()
else()
  # Set micromesh_python to be excluded from the build (default off)
  option(BUILD_MICROMESH_PYTHON "Require micromesh_python" OFF)

  if(BUILD_MICROMESH_PYTHON)
    # If the cached BUILD_MICROMESH_PYTHON was changed to ON, error out now
    message(FATAL_ERROR "Project ${PROJECT_NAME} is missing requirements (python and/or pybind11)")
  elseif(NOT (Python3_FOUND AND EXISTS ${PYBIND11_INCLUDE_DIR}))
    # BUILD_MICROMESH_PYTHON may also have been turned OFF by the user, but it's impossible to tell
    message(WARNING "Skipping project ${PROJECT_NAME} due to missing requirements (python and/or pybind11). Enable BUILD_MICROMESH_PYTHON to abort instead.")
  endif()
endif()

if((TARGET ${PROJECT_NAME}) AND (DEFINED USD_BIN_DIR))
  #
  # Configure schema USD plugin target
  #
  set(SCHEMA_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/schema)

  file(GLOB SCHEMA_PLUGIN_SOURCE_FILES ${SCHEMA_PLUGIN_SOURCE_DIR}/displacementMicromapAPI.*
                                       ${SCHEMA_PLUGIN_SOURCE_DIR}/tokens.*
                                       ${SCHEMA_PLUGIN_SOURCE_DIR}/api.h)

  set(SCHEMA_PLUGIN_NAME nvMicromesh)
  add_library(${SCHEMA_PLUGIN_NAME} SHARED ${SCHEMA_PLUGIN_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

  # USD doesn't appear to compile in debug CXX20
  # https://github.com/PixarAnimationStudios/USD/issues/2183
  set_target_properties(${SCHEMA_PLUGIN_NAME} PROPERTIES CXX_STANDARD 17)

  #add_compile_definitions(TBB_USE_DEBUG=0)
  
  # nv-usd is not using the C++11 ABI.
  # Otherwise, you'll get error messages about missing symbols revolving around std::string etc
  
  target_compile_definitions(${SCHEMA_PLUGIN_NAME} PRIVATE
      #-D_GLIBCXX_USE_CXX11_ABI=0
      -DPXR_PYTHON_ENABLED=1
      -DNVMICROMESH_EXPORTS
  )

  if (WIN32)
    # The /Zc:inline option strips out the "arch_ctor_<name>" symbols, so disable it.
    # See https://groups.google.com/g/usd-interest/c/nWm7u3B6CQk/m/OvIkOIyAAwAJ
    target_compile_options(${SCHEMA_PLUGIN_NAME} PRIVATE /permissive /bigobj /utf-8 /wd4996 $<$<NOT:$<CONFIG:Debug>>:/Zc:inline->)
  else()
    target_compile_options(${SCHEMA_PLUGIN_NAME} PRIVATE -fpermissive -Wno-deprecated-declarations -Wno-deprecated -Wl,--no-undefined -Wl,--no-allow-shlib-undefined)
  endif()

  target_include_directories(${SCHEMA_PLUGIN_NAME} PUBLIC
                             ${COMMON_DIR}
                             ${Python3_INCLUDE_DIRS}
                             ${USD_INCLUDE_DIR}
                             ${USD_BOOST_INCLUDE_DIR})
                             
  target_link_libraries(${SCHEMA_PLUGIN_NAME} PRIVATE ${PLATFORM_LIBRARIES} "${USD_LIBRARIES}" "${Python3_LIBRARIES}")
  target_link_directories(${SCHEMA_PLUGIN_NAME} PUBLIC "${USD_LIBRARY_DIR}" "${Python3_LIBRARY_DIRS}")
  
  if (WIN32)
    target_compile_definitions(${SCHEMA_PLUGIN_NAME} PRIVATE $<$<CONFIG:Debug>:TBB_USE_DEBUG=1> $<$<NOT:$<CONFIG:Debug>>:TBB_USE_DEBUG=0>)
  endif()
  
  set_target_properties(${SCHEMA_PLUGIN_NAME} PROPERTIES FOLDER usd_libs)
  _finalize_target(${SCHEMA_PLUGIN_NAME})

  #
  # Configure schema python wrapper target
  #
  file(GLOB SCHEMA_PY_WRAPPER_SOURCE_FILES ${SCHEMA_PLUGIN_SOURCE_DIR}/module.cpp ${SCHEMA_PLUGIN_SOURCE_DIR}/wrap*.cpp)

  set(SCHEMA_PY_WRAPPER_NAME ${SCHEMA_PLUGIN_NAME})
  set(SCHEMA_PY_WRAPPER_LIB_NAME _${SCHEMA_PY_WRAPPER_NAME})
  add_library(${SCHEMA_PY_WRAPPER_LIB_NAME} SHARED ${SCHEMA_PY_WRAPPER_SOURCE_FILES})

  set_target_properties(${SCHEMA_PY_WRAPPER_LIB_NAME} PROPERTIES PREFIX "")
  set_target_properties(${SCHEMA_PY_WRAPPER_LIB_NAME} PROPERTIES CXX_STANDARD 17)
  if (WIN32)
    set_target_properties(${SCHEMA_PY_WRAPPER_LIB_NAME} PROPERTIES SUFFIX ".pyd")
  endif()

  #${SCHEMA_PY_WRAPPER_NAME}

  target_compile_definitions(${SCHEMA_PY_WRAPPER_LIB_NAME} PRIVATE 
          #-D_GLIBCXX_USE_CXX11_ABI=0
          -DPXR_PYTHON_MODULES_ENABLED=1 
          -DMFB_PACKAGE_NAME=nvMicromesh
          -DMFB_ALT_PACKAGE_NAME=nvMicromesh
          -DMFB_PACKAGE_MODULE=nvMicromesh
          -DBOOST_ALL_DYN_LINK
          -DBOOST_ALL_NO_LIB
          -DBOOST_PYTHON_NO_PY_SIGNATURES
  )

  if (WIN32)
    target_compile_definitions(${SCHEMA_PY_WRAPPER_LIB_NAME} PRIVATE $<$<CONFIG:Debug>:TBB_USE_DEBUG=1> $<$<NOT:$<CONFIG:Debug>>:TBB_USE_DEBUG=0>)
  endif()

  if (WIN32)
    target_compile_options(${SCHEMA_PY_WRAPPER_LIB_NAME} PRIVATE /wd4996)
  else()
    target_compile_options(${SCHEMA_PY_WRAPPER_LIB_NAME} PRIVATE -Wno-deprecated-declarations -Wno-deprecated -Wl,--no-undefined -Wl,--no-allow-shlib-undefined)
  endif()
  
  target_include_directories(${SCHEMA_PY_WRAPPER_LIB_NAME} PUBLIC ${Python3_INCLUDE_DIRS} ${USD_INCLUDE_DIR} ${USD_BOOST_INCLUDE_DIR})

  target_link_libraries(${SCHEMA_PY_WRAPPER_LIB_NAME} PRIVATE ${PLATFORM_LIBRARIES} "${USD_LIBRARIES}" "${Python3_LIBRARIES}" "${SCHEMA_PLUGIN_NAME}")

  target_link_directories(${SCHEMA_PY_WRAPPER_LIB_NAME} PUBLIC "${USD_LIBRARY_DIR}" "${Python3_LIBRARY_DIRS}")

  set_target_properties(${SCHEMA_PY_WRAPPER_LIB_NAME} PROPERTIES FOLDER usd_libs)

  _finalize_target( ${SCHEMA_PY_WRAPPER_LIB_NAME} )

  #
  # Post-build steps for creating the nvMicromesh USD schema plugin
  #
  set(NVMICROMESH_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/schema/nvMicromesh)
  set(NVMICROMESH_PLUGIN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvMicromesh)
  file(MAKE_DIRECTORY ${NVMICROMESH_PLUGIN_BINARY_DIR})
  file(MAKE_DIRECTORY ${NVMICROMESH_PLUGIN_BINARY_DIR}/resources)
  
  if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
                        $<$<CONFIG:Debug>:${CMAKE_CURRENT_BINARY_DIR}/Debug/${PROJECT_NAME}.pyd> 
                        $<$<NOT:$<CONFIG:Debug>>:${CMAKE_CURRENT_BINARY_DIR}/Release/${PROJECT_NAME}.pyd>
                        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pyd
    )
    add_custom_command(TARGET ${SCHEMA_PLUGIN_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
                        $<$<CONFIG:Debug>:${OUTPUT_PATH}/Debug/${SCHEMA_PLUGIN_NAME}.dll> 
                        $<$<NOT:$<CONFIG:Debug>>:${OUTPUT_PATH}/Release/${SCHEMA_PLUGIN_NAME}.dll>
                        ${NVMICROMESH_PLUGIN_BINARY_DIR}/${SCHEMA_PLUGIN_NAME}.dll
    )
    add_custom_command(TARGET ${SCHEMA_PY_WRAPPER_LIB_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
                        $<$<CONFIG:Debug>:${OUTPUT_PATH}/Debug/${SCHEMA_PY_WRAPPER_LIB_NAME}.pyd> 
                        $<$<NOT:$<CONFIG:Debug>>:${OUTPUT_PATH}/Release/${SCHEMA_PY_WRAPPER_LIB_NAME}.pyd>
                        ${NVMICROMESH_PLUGIN_BINARY_DIR}/${SCHEMA_PY_WRAPPER_LIB_NAME}.pyd
      COMMAND ${CMAKE_COMMAND} -E copy
                        ${NVMICROMESH_PLUGIN_SOURCE_DIR}/__init__.py
                        ${NVMICROMESH_PLUGIN_BINARY_DIR}/__init__.py
    )
    add_custom_command(TARGET ${SCHEMA_PLUGIN_NAME} PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${NVMICROMESH_PLUGIN_SOURCE_DIR}/resources/plugInfo_windows-x86_64.json
                                       ${NVMICROMESH_PLUGIN_BINARY_DIR}/resources/plugInfo.json
      COMMAND ${CMAKE_COMMAND} -E copy ${NVMICROMESH_PLUGIN_SOURCE_DIR}/../generatedSchema.usda
                                       ${NVMICROMESH_PLUGIN_BINARY_DIR}/resources/generatedSchema.usda
    )
  elseif (UNIX)
    add_custom_command(TARGET ${SCHEMA_PLUGIN_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
                        $<$<CONFIG:Debug>:${OUTPUT_PATH}/Debug/lib${SCHEMA_PLUGIN_NAME}.so> 
                        $<$<NOT:$<CONFIG:Debug>>:${OUTPUT_PATH}/Release/lib${SCHEMA_PLUGIN_NAME}.so>
                        ${NVMICROMESH_PLUGIN_BINARY_DIR}/lib${SCHEMA_PLUGIN_NAME}.so
    )
    add_custom_command(TARGET ${SCHEMA_PY_WRAPPER_LIB_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
                        $<$<CONFIG:Debug>:${OUTPUT_PATH}/Debug/${SCHEMA_PY_WRAPPER_LIB_NAME}.so> 
                        $<$<NOT:$<CONFIG:Debug>>:${OUTPUT_PATH}/Release/${SCHEMA_PY_WRAPPER_LIB_NAME}.so>
                        ${NVMICROMESH_PLUGIN_BINARY_DIR}/${SCHEMA_PY_WRAPPER_LIB_NAME}.so
      COMMAND ${CMAKE_COMMAND} -E copy
                        ${NVMICROMESH_PLUGIN_SOURCE_DIR}/__init__.py
                        ${NVMICROMESH_PLUGIN_BINARY_DIR}/__init__.py
    )
  
    add_custom_command(TARGET ${SCHEMA_PLUGIN_NAME} PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${NVMICROMESH_PLUGIN_SOURCE_DIR}/resources/plugInfo_linux-x86_64.json
                                       ${NVMICROMESH_PLUGIN_BINARY_DIR}/resources/plugInfo.json
      COMMAND ${CMAKE_COMMAND} -E copy ${NVMICROMESH_PLUGIN_SOURCE_DIR}/../generatedSchema.usda
                                       ${NVMICROMESH_PLUGIN_BINARY_DIR}/resources/generatedSchema.usda
    )
  endif()


  #
  # genschema target to regenerate the USD schema headers and source files
  #
  # Note: after generating schema it is important to add the following to the NvMicromeshDisplacementMicromapAPI class declaration
  #  
  # class
  # +#if defined(ARCH_COMPILER_GCC) && ARCH_COMPILER_GCC_MAJOR >= 4 || defined(ARCH_COMPILER_CLANG)
  # +  NVMICROMESH_API
  # +#endif
  # NvMicromeshDisplacementMicromapAPI : public UsdAPISchemaBase
  # 
  set(SCHEMA_TARGET genschema)
  add_custom_target(${SCHEMA_TARGET})
  set_target_properties(${SCHEMA_TARGET} PROPERTIES FOLDER usd_libs)

  add_custom_command(TARGET ${SCHEMA_TARGET}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/python.cmd -m pip install jinja2
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/python.cmd ${USD_RELEASE_BIN_DIR}/usdGenSchema ${SCHEMA_PLUGIN_SOURCE_DIR}/schema.usda ${SCHEMA_PLUGIN_SOURCE_DIR}
  )
endif()

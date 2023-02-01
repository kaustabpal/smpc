cmake_minimum_required(VERSION 3.5)
project(acado_ros)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(acado_msgs REQUIRED)
find_package( ACADO REQUIRED )


add_library(mpc_solver
externals/qpoases/SRC/Bounds.cpp
  externals/qpoases/SRC/Constraints.cpp
  externals/qpoases/SRC/CyclingManager.cpp
  externals/qpoases/SRC/Indexlist.cpp
  externals/qpoases/SRC/MessageHandling.cpp
  externals/qpoases/SRC/QProblem.cpp
  externals/qpoases/SRC/QProblemB.cpp
  externals/qpoases/SRC/SubjectTo.cpp
  externals/qpoases/SRC/Utils.cpp
  externals/qpoases/SRC/EXTRAS/SolutionAnalysis.cpp
  model/codegen/acado_qpoases_interface.cpp
  model/codegen/acado_integrator.c
  model/codegen/acado_solver.c
  model/codegen/acado_auxiliary_functions.c)


target_include_directories(mpc_solver PUBLIC
model/codegen/
externals/qpoases
externals/qpoases/INCLUDE
externals/qpoases/SRC)

link_directories(${ACADO_ENV_LIBRARY_DIRS})

INCLUDE_DIRECTORIES( . ${ACADO_ENV_INCLUDE_DIRS} )

include_directories(
  include
  ${ACADO_ENV_INCLUDE_DIRS})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_package()

add_executable(combined_ngsim src/combined_ngsim.cpp)
target_link_libraries(combined_ngsim eigen-quadprog ${ACADO_ENV_SHARED_LIBRARIES} mpc_solver)
ament_target_dependencies(combined_ngsim rclcpp std_msgs nav_msgs acado_msgs ${ACADO_ENV_SHARED_LIBRARIES})

add_executable(combined_mpc_multi src/combined_mpc_multi.cpp)
target_link_libraries(combined_mpc_multi eigen-quadprog ${ACADO_ENV_SHARED_LIBRARIES} mpc_solver)
ament_target_dependencies(combined_mpc_multi rclcpp std_msgs nav_msgs acado_msgs ${ACADO_ENV_SHARED_LIBRARIES})


install(TARGETS
  combined_ngsim
  combined_mpc_multi
  DESTINATION lib/${PROJECT_NAME})

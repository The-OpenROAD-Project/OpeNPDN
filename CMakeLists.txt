# Copyright (c) 2019, Parallax Software, Inc.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Setting CMake version properly to ensure correct policies behavior
cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
project(OpeNPDN)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(OPENPDN_HOME ${PROJECT_SOURCE_DIR})
set(OPENDB_HOME ${PROJECT_SOURCE_DIR}/modules/OpenDB)

find_package(TCL REQUIRED)
set(BUILD_PYTHON ON)
set(BUILD_TCL ON)
add_subdirectory(${OPENDB_HOME} ${CMAKE_CURRENT_BINARY_DIR}/modules/OpenDB)


cmake_minimum_required(VERSION 3.12)
project(stimfit_CI)

foreach(_dist suse ubuntu centos)
    add_custom_target(${_dist} ALL
        COMMAND docker build 
            -t stimfit:${_dist}  
            -f .ci/Dockerfile.${_dist}   
            ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
        COMMENT "Running docker build for ${_dist}"
        VERBATIM)
endforeach()

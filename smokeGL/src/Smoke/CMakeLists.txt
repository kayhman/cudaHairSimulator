include_directories(../smokeLib)
add_executable(smoke smoke.cpp objLoader.cpp objLoader.h)
target_link_libraries(smoke  smokeLib ${OPENGL_LIBRARIES})

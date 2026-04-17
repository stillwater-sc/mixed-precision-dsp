add_executable(ekf_bearing_range ekf_bearing_range.cpp)
target_link_libraries(ekf_bearing_range PRIVATE sw::dsp)

add_executable(ukf_tracking ukf_tracking.cpp)
target_link_libraries(ukf_tracking PRIVATE sw::dsp)

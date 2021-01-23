
add_library(jetracer_library
    Events/EventTypes.cpp
    Fault.cpp
    EventsThread.cpp
    MainEventsLoop.cpp
    PingPong.cpp
    RealSense/RealSenseD400.cpp
)


#ds main executable (pthread is used for the GUI thread)
add_executable(jetracer main.cpp)
target_link_libraries(jetracer jetracer_library -pthread)


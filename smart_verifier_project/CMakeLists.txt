cmake_minimum_required(VERSION 3.10)
project(smart_verifier C)

set(CMAKE_C_STANDARD 11)

include_directories(
    src
    src/lvgl
    src/tflite
)

add_executable(smart_verifier
    src/main.c
    src/ui.c
)

target_link_libraries(smart_verifier
    tensorflow-lite
    pthread
    rt
    dl
)

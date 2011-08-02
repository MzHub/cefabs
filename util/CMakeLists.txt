INCLUDE(${QT_USE_FILE})

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/gpu )

SET(sources 
    cudadevicedialog.cpp
    cudadevicedialog.h
    cudadevicedialog.ui
    fancystyle.cpp
    fancystyle.h
    imageutil.cpp
    imageutil.h
    imageview.cpp
    imageview.h
    param.cpp
    param.h
    paramui.cpp
    paramui.h
    rolloutbox.cpp
    rolloutbox.h
    stable.cpp
    stable.h
    videoplayer.cpp
    videoplayer.h
    videocontrols.h
    videocontrols.cpp )

IF(FFMPEG_FOUND)
    INCLUDE_DIRECTORIES( ${FFmpeg_INCLUDE_DIR} )
    SET(sources 
        ${sources}
        ffmpeg_decoder.cpp
        ffmpeg_decoder.h
        ffmpeg_encoder.cpp
        ffmpeg_encoder.h )
ENDIF()

QT4_AUTO_WRAP(sources ${sources})
CUDA_ADD_LIBRARY(util STATIC ${sources})

SOURCE_GROUP(src REGULAR_EXPRESSION "c$|cpp$|hpp$|h$|ui$|qrc$|cu$")
SOURCE_GROUP(generated REGULAR_EXPRESSION "cxx$|ui_")

IF(MSVC)
    IF(MSVC_IDE)
        SET_TARGET_PROPERTIES(util
            PROPERTIES
            COMPILE_FLAGS "/FIstable.h /Yustable.h")
            
        SET_SOURCE_FILES_PROPERTIES(stable.cpp
            PROPERTIES
            COMPILE_FLAGS "/Ycstable.h")
    ELSE()
        SET_TARGET_PROPERTIES(util
            PROPERTIES
            COMPILE_FLAGS "/FIstable.h")
    ENDIF()
ELSE(MSVC)
    SET_TARGET_PROPERTIES(util
        PROPERTIES
        COMPILE_FLAGS "-include stable.h")
ENDIF(MSVC)

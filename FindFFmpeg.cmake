#  FFMPEG_FOUND - system has FFmpeg
#  FFmpeg_DIR - the FFmpeg directory
#  FFmpeg_INCLUDE_DIR - the FFmpeg include directory
#  FFmpeg_LIBRARIES - link these to use NPP

IF(CMAKE_CL_64)
    FIND_PATH(FFmpeg_DIR include/libavformat/avformat.h
        PATHS "${PROJECT_SOURCE_DIR}/../3rdparty/x64")
ELSE()
    FIND_PATH(FFmpeg_DIR include/libavformat/avformat.h
        PATHS "${PROJECT_SOURCE_DIR}/../3rdparty/win32")
ENDIF()

FIND_PATH(FFmpeg_INCLUDE_DIR libavformat/avformat.h HINTS "${FFmpeg_DIR}/include")

FIND_LIBRARY(FFmpeg_avutil_LIBRARY avutil HINTS "${FFmpeg_DIR}/lib")
FIND_LIBRARY(FFmpeg_avcodec_LIBRARY avcodec HINTS "${FFmpeg_DIR}/lib")
FIND_LIBRARY(FFmpeg_avformat_LIBRARY avformat HINTS "${FFmpeg_DIR}/lib")
FIND_LIBRARY(FFmpeg_swscale_LIBRARY swscale HINTS "${FFmpeg_DIR}/lib")

IF(WIN32)
    FIND_PROGRAM(FFmpeg_avutil_DLL avutil.dll NAMES avutil-51.dll HINTS ${FFmpeg_DIR}/bin )
    FIND_PROGRAM(FFmpeg_avcodec_DLL avcodec.dll NAMES avcodec-53.dll HINTS ${FFmpeg_DIR}/bin )
    FIND_PROGRAM(FFmpeg_avformat_DLL avformat.dll NAMES avformat-53.dll HINTS ${FFmpeg_DIR}/bin )
    FIND_PROGRAM(FFmpeg_swscale_DLL swscale.dll NAMES swscale-2.dll HINTS ${FFmpeg_DIR}/bin )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFmpeg DEFAULT_MSG 
        FFmpeg_INCLUDE_DIR
        FFmpeg_avutil_LIBRARY FFmpeg_avutil_DLL
        FFmpeg_avcodec_LIBRARY FFmpeg_avcodec_DLL
        FFmpeg_avformat_LIBRARY FFmpeg_avformat_DLL
        FFmpeg_swscale_LIBRARY FFmpeg_swscale_DLL )
ELSE()
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFmpeg DEFAULT_MSG 
        FFmpeg_INCLUDE_DIR
        FFmpeg_avcodec_LIBRARY
        FFmpeg_avformat_LIBRARY
        FFmpeg_avutil_LIBRARY
        FFmpeg_swscale_LIBRARY )
ENDIF()

IF(FFMPEG_FOUND)
    SET(FFmpeg_LIBRARIES
        ${FFmpeg_avcodec_LIBRARY}
        ${FFmpeg_avformat_LIBRARY}
        ${FFmpeg_avutil_LIBRARY}
        ${FFmpeg_swscale_LIBRARY} )
    IF(APPLE)
        SET(FFmpeg_LIBRARIES ${FFmpeg_LIBRARIES} -lx264 -lbz2 -lz)
    ENDIF()
    MARK_AS_ADVANCED(FFmpeg_LIBRARIES)
ENDIF()

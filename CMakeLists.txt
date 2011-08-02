CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cefabs)

SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}" )
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

IF(EXISTS "${PROJECT_SOURCE_DIR}/.svn")
    IF(APPLE)
        SET(CPACK_GENERATOR "DragNDrop")
    ELSE()
        SET(CPACK_GENERATOR "ZIP")
    ENDIF()
    SET(CPACK_PACKAGE_VERSION_MAJOR "0")
    SET(CPACK_PACKAGE_VERSION_MINOR "1")
    SET(CPACK_PACKAGE_VERSION_PATCH "0")
    IF(WIN32)
        FIND_PROGRAM(SubWCRev_PROG SubWCRev)
        MARK_AS_ADVANCED(SubWCRev_PROG)
        IF(SubWCRev_PROG)
            EXECUTE_PROCESS(
                COMMAND ${SubWCRev_PROG} ${PROJECT_SOURCE_DIR} -f
                OUTPUT_VARIABLE SubWCRev_OUTPUT
                RESULT_VARIABLE SubWCRev_RESULT)
            IF(${SubWCRev_RESULT} EQUAL 0)
                STRING(REGEX REPLACE ".*Last committed at revision ([0-9]+).*" "\\1" BUILD_REV "${SubWCRev_OUTPUT}")
                MESSAGE(STATUS "BUILD_REV: ${BUILD_REV}")
                SET(CPACK_PACKAGE_VERSION_PATCH "${BUILD_REV}")
            ELSE()
                MESSAGE(WARNING ${SubWCRev_OUTPUT})
            ENDIF()
        ENDIF()
    ELSE()
        FIND_PROGRAM(svnversion_PROG svnversion)
        MARK_AS_ADVANCED(svnversion_PROG)
        IF(svnversion_PROG)
            EXECUTE_PROCESS(
                COMMAND ${svnversion_PROG} -n ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE svnversion_OUTPUT
                RESULT_VARIABLE svnversion_RESULT)
            IF(${svnversion_RESULT} EQUAL 0)
                MESSAGE(STATUS "BUILD_REV: ${svnversion_OUTPUT}")
                SET(CPACK_PACKAGE_VERSION_PATCH "${svnversion_OUTPUT}")
            ELSE()
                MESSAGE(WARNING ${svnversion_OUTPUT})
            ENDIF()
        ENDIF()
    ENDIF()
    SET(CPACK_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
    IF(${CPACK_SYSTEM_NAME} MATCHES Windows)
        IF(CMAKE_CL_64)
            SET(CPACK_SYSTEM_NAME win64)
        ELSE()
            SET(CPACK_SYSTEM_NAME win32)
        ENDIF()
    ENDIF()
    IF(${CPACK_SYSTEM_NAME} MATCHES Darwin)
        SET(CPACK_SYSTEM_NAME mac)
    ENDIF()
    SET(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-r${CPACK_PACKAGE_VERSION_PATCH}-${CPACK_SYSTEM_NAME}")
    SET(CPACK_SOURCE_PACKAGE_FILE_NAME "${PROJECT_NAME}-r${CPACK_PACKAGE_VERSION_PATCH}-src")
    SET(CPACK_SOURCE_IGNORE_FILES "/build/;/.svn/")
    INCLUDE(CPack)
ENDIF()

IF(MSVC)
    ADD_DEFINITIONS(/D_CRT_SECURE_NO_WARNINGS /D_STATIC_CPPLIB)
ENDIF(MSVC)

SET(CUDA_ARCH "sm_11" CACHE STRING "CUDA architecture")
SET_PROPERTY(CACHE CUDA_ARCH PROPERTY STRINGS sm_11 sm_20 sm_21 none)
IF(NOT (${CUDA_ARCH} STREQUAL none))
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};;-arch=${CUDA_ARCH}")
ENDIF()
OPTION(CUDA_PTXAS_VERBOSE "Show ptxas verbose information" OFF)
IF(${CUDA_PTXAS_VERBOSE})
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};;--ptxas-options=-v")
ENDIF()
FIND_PACKAGE(CUDA 4.0 REQUIRED)

FIND_PACKAGE(FFmpeg)
IF(FFMPEG_FOUND)
    ADD_DEFINITIONS(/DHAVE_FFMPEG)
ENDIF()

FIND_PACKAGE(Qt4)

MACRO(QT4_AUTO_WRAP outfiles)
    FOREACH(fileName ${ARGN})
        IF(fileName MATCHES "\\.h$")
            FILE(STRINGS ${fileName} lines REGEX Q_OBJECT)
            IF(lines)
                SET(moc_headers ${moc_headers} ${fileName})
                MESSAGE(STATUS "moc: ${fileName}")
            ENDIF()
        ENDIF()
        IF(fileName MATCHES "\\.ui$")
            SET(ui_files ${ui_files} ${fileName})
            MESSAGE(STATUS "uic: ${fileName}")
        ENDIF()
        IF(fileName MATCHES "\\.qrc$")
            SET(qrc_files ${qrc_files} ${fileName})
            MESSAGE(STATUS "qrc: ${fileName}")
        ENDIF()
    ENDFOREACH()
    QT4_WRAP_CPP(${outfiles} ${moc_headers})
    QT4_WRAP_UI(${outfiles} ${ui_files})
    QT4_ADD_RESOURCES(${outfiles} ${qrc_files})
ENDMACRO()

MACRO(FIXUP_BUNDLE targetName)
    IF(APPLE)
        SET(APP "\${CMAKE_INSTALL_PREFIX}/${targetName}.app")
        SET(DIRS ${QT_LIBRARY_DIRS})
        INSTALL(CODE "
            file(WRITE \"${APP}/Contents/Resources/qt.conf\" \"\")
            include(BundleUtilities)
            function(gp_resolve_item_override context item exepath dirs resolved_item_var resolved_var)
                IF (\${item} STREQUAL \"@rpath/libcudart.dylib\")
                    #message(\"RI: \${item} \${\${resolved_item_var}} \${\${resolved_var}}\")
                    set(\${resolved_item_var} \"/usr/local/cuda/lib/libcudart.dylib\" PARENT_SCOPE)
                    set(\${resolved_var} 1 PARENT_SCOPE)
                ENDIF()
                IF (\${item} STREQUAL \"@rpath/libtlshook.dylib\")
                    #message(\"RI: \${item} \${\${resolved_item_var}} \${\${resolved_var}}\")
                    set(\${resolved_item_var} \"/usr/local/cuda/lib/libtlshook.dylib\" PARENT_SCOPE)
                    set(\${resolved_var} 1 PARENT_SCOPE)
                ENDIF()
            endfunction()
            function(gp_resolved_file_type_override file type_var)
                IF(\${file} MATCHES \"(CUDA.framework|libcuda.dylib)\")
                    #message(\"GP: \${file} \${\${type_var}}\")
                    set(\${type_var} \"system\" PARENT_SCOPE)
                ENDIF()
            endfunction()
            fixup_bundle(\"${APP}\" \"\" \"${DIRS}\")
            " COMPONENT Runtime)

    ENDIF()
ENDMACRO()

IF(WIN32)
    IF(CMAKE_CL_64)
        FILE(GLOB CUDA_CUDART_DLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64*.dll")
    ELSE()
        FILE(GLOB CUDA_CUDART_DLL "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart32*.dll")
    ENDIF()
    MARK_AS_ADVANCED(CUDA_CUDART_DLL)

    INSTALL(FILES 
        "${CUDA_CUDART_DLL}" 
        DESTINATION . 
        CONFIGURATIONS Debug Release)

    INSTALL(FILES 
        "${QT_BINARY_DIR}/QtCored4.dll" "${QT_BINARY_DIR}/QtGuid4.dll"
        DESTINATION . 
        CONFIGURATIONS Debug)

    INSTALL(FILES 
        "${QT_BINARY_DIR}/QtCore4.dll" "${QT_BINARY_DIR}/QtGui4.dll"
        DESTINATION . 
        CONFIGURATIONS Release)

	IF(EXISTS "${PROJECT_SOURCE_DIR}/.svn")
		ADD_CUSTOM_TARGET(PACKAGE_SOURCE
			COMMAND "${CMAKE_CPACK_COMMAND}" -C $(OutDir) --config "CPackSourceConfig.cmake"
			WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" )
	ENDIF()
	
    ADD_CUSTOM_TARGET(STAGE ALL)
    IF(MSVC_IDE)
        SET(STAGE_CFG "$(ConfigurationName)")
    ELSE()
        SET(STAGE_CFG "${CMAKE_BUILD_TYPE}")
    ENDIF()
    SET(STAGE_DIR "bin/${STAGE_CFG}")
    ADD_CUSTOM_COMMAND(TARGET STAGE POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory ${STAGE_DIR}
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CUDA_CUDART_DLL}" ${STAGE_DIR}
        COMMAND if "${STAGE_CFG}"==Debug (
                "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtCored4.dll ${STAGE_DIR} &&
                "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtCored4.pdb ${STAGE_DIR} &&
                "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtGuid4.dll ${STAGE_DIR} && 
                "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtGuid4.pdb ${STAGE_DIR}
                ) else (
                "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtCore4.dll ${STAGE_DIR} &&
                "${CMAKE_COMMAND}" -E copy_if_different ${QT_BINARY_DIR}/QtGui4.dll ${STAGE_DIR}
                )
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    OPTION(INSTALL_MSVCRT "Install Microsoft CRT" OFF)
    IF(${INSTALL_MSVCRT})
        SET(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ".")
        INCLUDE(InstallRequiredSystemLibraries)
    ENDIF()
    
    IF(FFMPEG_FOUND)
        INSTALL(FILES 
            ${FFmpeg_avutil_DLL} ${FFmpeg_avcodec_DLL} ${FFmpeg_avformat_DLL} ${FFmpeg_swscale_DLL}   
            DESTINATION . 
            CONFIGURATIONS Debug Release)

        ADD_CUSTOM_COMMAND(TARGET STAGE POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${FFmpeg_avutil_DLL}" ${STAGE_DIR} &&
                    "${CMAKE_COMMAND}" -E copy_if_different "${FFmpeg_avcodec_DLL}" ${STAGE_DIR} &&
                    "${CMAKE_COMMAND}" -E copy_if_different "${FFmpeg_avformat_DLL}" ${STAGE_DIR} &&
                    "${CMAKE_COMMAND}" -E copy_if_different "${FFmpeg_swscale_DLL}" ${STAGE_DIR}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    ENDIF()
    
ENDIF()

ADD_SUBDIRECTORY(gpu)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(util)

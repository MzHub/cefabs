@echo off
cd /d %~dp0
cmake -E make_directory build || exit /B 1
cd build
if "%1"=="x64" ( 
    cmake -G "Visual Studio 9 2008 Win64" .. || exit /B 1
) else (
    cmake -G "Visual Studio 9 2008" .. || exit /B 1
) 
devenv /build Release cefabs.sln || exit /B 1
cpack || exit /B 1
cpack --config CPackSourceConfig.cmake || exit /B 1
cd /d %~dp0

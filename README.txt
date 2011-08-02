//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2011 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 

Requirements:

    * CUDA 4: http://developer.nvidia.com/cuda-toolkit-40
    * CMake: http://www.cmake.org
    * Qt: http://qt.nokia.com
    * FFmpeg: http://www.ffmpeg.org [optional]

Building:

    Windows / Visual Studio:
        1) mkdir build
        2) cd build
        3) cmake ..
        4) devenv /build Release cefabs.sln
    
    Mac OS X / Linux:
        1) mkdir build
        2) cd build
        3) cmake ..
        4) make


Precompiled binaries for Windows require the Visual Studio 2008 runtimes:
    
    * Microsoft Visual C++ 2008 SP1 Redistributable Package (x86):
      http://www.microsoft.com/downloads/en/details.aspx?familyid=A5C84275-3B97-4AB7-A40D-3802B2AF5FC2
      
    * Microsoft Visual C++ 2008 SP1 Redistributable Package (x64):
      http://www.microsoft.com/downloads/en/details.aspx?FamilyID=BA9257CA-337F-4B40-8C14-157CFDFFEE4E
 

 Related Publications:

    * Kyprianidis, J. E., & Kang, H. (2011). Image and Video 
      Abstraction by Coherence-Enhancing Filtering. Computer Graphics 
      Forum, 30(2), 593-602. 

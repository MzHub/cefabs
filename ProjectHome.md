This program implements an automatic image and video abstraction technique based on adaptive line integral convolution and directional shock filtering. It was written by [Jan Eric Kyprianidis](http://www.kyprianidis.com) while working as a research scientist for the [computer graphics systems group](http://www.hpi3d.de) of the [Hasso-Plattner-Institut](http://www.hpi.uni-potsdam.de) at the University of Potsdam, Germany.

![http://wiki.cefabs.googlecode.com/git/screenshot.jpg](http://wiki.cefabs.googlecode.com/git/screenshot.jpg)

## Building ##

Building requires [CMake](http://www.cmake.org), [CUDA](http://developer.nvidia.com/cuda-toolkit-40), and the [Qt cross platform toolkit](http://qt.nokia.com). Recommended CUDA version is >= 4.0 and recommended Qt version is 4.7.3. The program has been tested to successfully build with Visual Studio 2008 on Windows, Qt Creator 2.2.1 on Mac OS X and the default toolchain on Ubuntu 11.04. See build.bat/build.sh to get started. Video processing requires [FFmpeg](http://www.ffmpeg.org), but is optional.

## Related Publications ##

  * [Kyprianidis, J. E.](http://www.kyprianidis.com), & [Kang, H.](http://www.cs.umsl.edu/~kang/) (2011). [Image and Video Abstraction by Coherence-Enhancing Filtering](http://www.kyprianidis.com/p/eg2011). _Computer Graphics Forum_ 30(2), pp. 593-602. (Proceedings Eurographics 2011)
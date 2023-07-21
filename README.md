# Cpp_MNIST_OCR_NN_in_a_weekend__MWE
This project is a Minimal Working Example (MWE) of a Neural Network OCR program for the MNIST data set, written in C++.
## Purpose & Credit
As a personal exercise I have created this repository with the purpose of building a Neural Network OCR from scratch in C++. As my main guide I've followed the blogpost [C++ Neural Network in a Weekend](https://www.jeremyong.com/cpp/machine-learning/2020/10/23/cpp-neural-network-in-a-weekend/)* by Jeremy Ong. It seemed like a very thouruohg and well written walkthrough, much credit to Mr. Ong.

*PDF version: https://raw.githubusercontent.com/jeremyong/cpp_nn_in_a_weekend/master/doc/DOC.pdf

## How to compile and run
### Prerequisites
 - C++ compiler (e.g. g++ or Clang), with C++11 support
 - CMake version 3.8 or higher
 - Ninja build system (for CMake)

### Build the project
```
git clone https://github.com/VictorieeMan/Cpp_MNIST_OCR_NN_in_a_weekend__MWE.git
```
Easiest way to build is by using Visual Studio. Just git clone this repo, and open the root folder with Visual Studio, then build and compile. This project is however configured with **CMake** files, that can be used to build & compile the project without Visual Studio. See below for instructions.

Super commands for building the project using CMake:

Windows:
```
git clone https://github.com/VictorieeMan/Cpp_MNIST_OCR_NN_in_a_weekend__MWE.git && cd .\Cpp_MNIST_OCR_NN_in_a_weekend__MWE\ && cmake --preset=x64-debug -B build -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ && cmake --build build
```

UNIX:
```	
git clone https://github.com/VictorieeMan/Cpp_MNIST_OCR_NN_in_a_weekend__MWE.git && cd Cpp_MNIST_OCR_NN_in_a_weekend__MWE/ && cmake --preset=unix-debug -B build -G "Unix Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ && cmake --build build && cd build && main.exe
```
**NOTE:** For customization of the command, -B is the build directory, -G is the generator, -DCMAKE_C_COMPILER and -DCMAKE_CXX_COMPILER are the compilers to use. You may alter these variables if the presets above doesn't match with your system.

#### Super commands explained
```
git clone 
```
Clones the repository to your local machine.

```
cd .\Cpp_MNIST_OCR_NN_in_a_weekend__MWE\
```
Changes directory to the root folder of the project.

```
//Windows version
cmake --preset=x64-debug -B build -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

//UNIX version
cmake --preset=unix-debug -B build -G "Unix Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
```
Creates a build directory called "build" and generates the build files using the Ninja generator. The compiler is set to Clang and Clang++, alter these variables if the presets above doesn't match with your system.

```
cmake --build build
```
Builds the project using the build files generated in the previous step.

```
cd build
```
Changes directory to the build directory.

```
main.exe
```
Runs the program.


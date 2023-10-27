# CVOps Inference

A C++ library with exposed C API for cross-platform inference of computer vision models.  Designed to be wrapped by Python Ctypes and Dart's FFI modules.

## Installation Requirements:

* Linux (Ubuntu)

```bash
sudo snap install cmake --classic
sudo apt install libopencv-dev libboost-all-dev libjsoncpp-dev
```

## To Build

```bash
cmake .
cmake --build .
```
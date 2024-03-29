#pragma once

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

void writeImage(const char* path, int w, int h, uint8_t* data);

void writeImageFromDevice(const char* path, int w, int h, hiprtDevicePtr data);
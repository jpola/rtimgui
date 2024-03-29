#include "ImageWriter.h"
#include <stb_image_write.h>
#include "assert.h"

void writeImage(const char* path, int w, int h, uint8_t* data)
{
    stbi_write_png(path, w, h, 4, data, w * 4);
}

void writeImageFromDevice(const char* path, int w, int h, hiprtDevicePtr data)
{
    uint8_t* tmp = new uint8_t[w * h * 4];
    uint8_t* tmp1 = new uint8_t[w * h * 4];

    HIP_ASSERT(hipMemcpyDtoH(tmp, data, w * h * 4) == hipSuccess, "copy");

    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
        {
            int idx = i + j * w;
            int dIdx = i + (h - 1 - j) * w;
            for (int k = 0; k < 4; k++) tmp1[dIdx * 4 + k] = tmp[idx * 4 + k];
        }
    writeImage(path, w, h, tmp1);
    delete[] tmp;
    delete[] tmp1;
};
#include "constant.h"


__kernel void nothing(__global float* dest, __global float* source)
{
    
}


__kernel void gpu_memcpy(__global float4* dest, __global float4* source)
{
    int x = get_global_id(0);
    int size = get_global_size(0);
    dest[x] = source[x];
    dest[x + size] = source[x + size];
    dest[x + size * 2] = source[x + size * 2];
    dest[x + size * 3] = source[x + size * 3];

}

__kernel void gpu_memcpy_2(__global float2* dest, __global float2* source)
{
    int x = get_global_id(0);
    int size = get_global_size(0);
    dest[x] = source[x];
    dest[x + size] = source[x + size];
    dest[x + size * 2] = source[x + size * 2];
    dest[x + size * 3] = source[x + size * 3];
    dest[x + size * 4] = source[x + size * 4];
    dest[x + size * 5] = source[x + size * 5];
    dest[x + size * 6] = source[x + size * 6];
    dest[x + size * 7] = source[x + size * 7];
}

__kernel void gpu_memcpy_gather(__global float* dest, __global float* source)
{
    int x = get_global_id(0);
    int size = get_global_size(0);
    dest[x] = source[x];
    dest[x + size] = source[x + size];
    dest[x + size * 2] = source[x + size * 2];
    dest[x + size * 3] = source[x + size * 3];
    dest[x + size * 4] = source[x + size * 4];
    dest[x + size * 5] = source[x + size * 5];
    dest[x + size * 6] = source[x + size * 6];
    dest[x + size * 7] = source[x + size * 7];
    dest[x + size * 8] = source[x + size * 8];
    dest[x + size * 9] = source[x + size * 9];
    dest[x + size * 10] = source[x + size * 10];
    dest[x + size * 11] = source[x + size * 11];
    dest[x + size * 12] = source[x + size * 12];
    dest[x + size * 13] = source[x + size * 13];
    dest[x + size * 14] = source[x + size * 14];
    dest[x + size * 15] = source[x + size * 15];

}


__kernel void gpu_memcpy_tex_RGBA(__global float* dest, __read_only image2d_t source)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int x = get_global_id(0);
    int rgbx = x >> 2;
    int2 coord;
    coord.x = rgbx & WIDTH_MASK;
    coord.y = rgbx >> LOG_WIDTH;
    float4 ans = read_imagef(source, smp, coord);
    int rgbid = x % 4;
    int finalans = ans.x;
    if (rgbid == 1)
        finalans = ans.y;
    else if (rgbid == 2)
        finalans = ans.z;
    else if (rgbid == 3)
        finalans = ans.w;
    dest[x] = finalans;
}

__kernel void gpu_memcpy_tex_RGBA4(__global float4* dest, __read_only image2d_t source)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int x = get_global_id(0);
    int2 coord;
    coord.x = x & WIDTH_MASK;
    coord.y = x >> LOG_WIDTH;
    float4 ans = read_imagef(source, smp, coord);
    dest[x] = ans;
}

__kernel void gpu_memcpy_tex_R(__global float* dest, __read_only image2d_t source)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int x = get_global_id(0);
    int2 coord;
    coord.x = x & WIDTH_MASK;
    coord.y = x >> LOG_WIDTH;
    float4 ans = read_imagef(source, smp, coord);
    dest[x] = ans.x;
}


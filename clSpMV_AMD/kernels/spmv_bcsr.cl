#include "constant.h"

//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_bcsr_red_14(__global int* rowptr,  __global int* colid, __global float4* data, __global float4* vec, __global float* result, int data_align)
{
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    float4 matelem = data[i];
	    float4 vecelem = vec[col];
            sum = mad(matelem, vecelem, sum);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x = sum.x + sum.y + sum.z + sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum.x;
            
    }
}


__kernel void gpu_bcsr_red_14_tx(__global int* rowptr,  __global int* colid, __global float4* data, __read_only image2d_t vec, __global float* result, int data_align)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (row < row_num)
    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
	    float4 matelem = data[i];
	    float4 vecelem = read_imagef(vec, smp, coord);
            sum = mad(matelem, vecelem, sum);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x = sum.x + sum.y + sum.z + sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum.x;
            
    }
}

//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_bcsr_red_24(__global int* rowptr,  __global int* colid, __global float4* data, __global float4* vec, __global float* result, int data_align)
{
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = vec[col];
            sum = mad(matelem, vecelem, sum);
	    matoffset += data_align;
	    matelem = data[matoffset];
            sum2 = mad(matelem, vecelem, sum2);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x = sum.x + sum.y + sum.z + sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2] += sum.x;
        
	s_ptr[localid] = sum2.x = sum2.x + sum2.y + sum2.z + sum2.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 32];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 16];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 8];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 4];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 2];
        sum2.x = sum2.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2 + 1] += sum2.x;
            
            
    }
}


__kernel void gpu_bcsr_red_24_tx(__global int* rowptr,  __global int* colid, __global float4* data, __read_only image2d_t vec, __global float* result, int data_align)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (row < row_num)
    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    int matoffset = i;
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
	    float4 matelem = data[matoffset];
	    float4 vecelem = read_imagef(vec, smp, coord);
            sum = mad(matelem, vecelem, sum);
	    matoffset += data_align;
	    matelem = data[matoffset];
            sum2 = mad(matelem, vecelem, sum2);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x = sum.x + sum.y + sum.z + sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2] += sum.x;
        
	s_ptr[localid] = sum2.x = sum2.x + sum2.y + sum2.z + sum2.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 32];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 16];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 8];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 4];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 2];
        sum2.x = sum2.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2 + 1] += sum2.x;
    }
}

//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_bcsr_red_44(__global int* rowptr,  __global int* colid, __global float4* data, __global float4* vec, __global float4* result, int data_align)
{
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = vec[col];
            float4 tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        s_ptr[localid] = sum.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 32];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 16];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 8];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 4];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 2];
        sum.y = sum.y + s_ptr[localid + 1];

        s_ptr[localid] = sum.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 32];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 16];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 8];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 4];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 2];
        sum.z = sum.z + s_ptr[localid + 1];

        s_ptr[localid] = sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 32];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 16];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 8];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 4];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 2];
        sum.w = sum.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum;
            
    }
}


__kernel void gpu_bcsr_red_44_tx(__global int* rowptr,  __global int* colid, __global float4* data, __read_only image2d_t vec, __global float4* result, int data_align)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (row < row_num)
    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = read_imagef(vec, smp, coord);
	    float4 tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
        }

	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        s_ptr[localid] = sum.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 32];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 16];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 8];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 4];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 2];
        sum.y = sum.y + s_ptr[localid + 1];

        s_ptr[localid] = sum.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 32];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 16];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 8];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 4];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 2];
        sum.z = sum.z + s_ptr[localid + 1];

        s_ptr[localid] = sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 32];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 16];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 8];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 4];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 2];
        sum.w = sum.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum;
            
    }
}

//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_bcsr_red_84(__global int* rowptr,  __global int* colid, __global float4* data, __global float4* vec, __global float4* result, int data_align)
{
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = vec[col];
            float4 tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.x = sum2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.y = sum2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.z = sum2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.w = sum2.w + tmp.x + tmp.y + tmp.z + tmp.w;
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        s_ptr[localid] = sum.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 32];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 16];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 8];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 4];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 2];
        sum.y = sum.y + s_ptr[localid + 1];

        s_ptr[localid] = sum.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 32];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 16];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 8];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 4];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 2];
        sum.z = sum.z + s_ptr[localid + 1];

        s_ptr[localid] = sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 32];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 16];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 8];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 4];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 2];
        sum.w = sum.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2] += sum;
            
        s_ptr[localid] = sum2.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 32];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 16];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 8];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 4];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 2];
        sum2.x = sum2.x + s_ptr[localid + 1];

        s_ptr[localid] = sum2.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 32];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 16];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 8];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 4];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 2];
        sum2.y = sum2.y + s_ptr[localid + 1];

        s_ptr[localid] = sum2.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 32];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 16];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 8];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 4];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 2];
        sum2.z = sum2.z + s_ptr[localid + 1];

        s_ptr[localid] = sum2.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 32];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 16];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 8];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 4];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 2];
        sum2.w = sum2.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2 + 1] += sum2;
            
    }
}


__kernel void gpu_bcsr_red_84_tx(__global int* rowptr,  __global int* colid, __global float4* data, __read_only image2d_t vec, __global float4* result, int data_align)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (row < row_num)
    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = read_imagef(vec, smp, coord);
	    float4 tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.x = sum2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.y = sum2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.z = sum2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.w = sum2.w + tmp.x + tmp.y + tmp.z + tmp.w;
        }

	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        s_ptr[localid] = sum.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 32];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 16];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 8];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 4];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 2];
        sum.y = sum.y + s_ptr[localid + 1];

        s_ptr[localid] = sum.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 32];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 16];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 8];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 4];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 2];
        sum.z = sum.z + s_ptr[localid + 1];

        s_ptr[localid] = sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 32];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 16];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 8];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 4];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 2];
        sum.w = sum.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2] += sum;
            
        s_ptr[localid] = sum2.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 32];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 16];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 8];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 4];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 2];
        sum2.x = sum2.x + s_ptr[localid + 1];

        s_ptr[localid] = sum2.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 32];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 16];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 8];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 4];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 2];
        sum2.y = sum2.y + s_ptr[localid + 1];

        s_ptr[localid] = sum2.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 32];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 16];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 8];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 4];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 2];
        sum2.z = sum2.z + s_ptr[localid + 1];

        s_ptr[localid] = sum2.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 32];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 16];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 8];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 4];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 2];
        sum2.w = sum2.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2 + 1] += sum2;
            
    }
}


//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_bcsr_red_18(__global int* rowptr,  __global int* colid, __global float4* data, __global float4* vec, __global float* result, int data_align)
{
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    float4 matelem = data[i];
	    float4 vecelem = vec[col];
            sum = mad(matelem, vecelem, sum);
	    
	    vecelem = vec[col + 1];
	    matelem = data[i + data_align];
            sum = mad(matelem, vecelem, sum);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x = sum.x + sum.y + sum.z + sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum.x;
            
    }
}


__kernel void gpu_bcsr_red_18_tx(__global int* rowptr,  __global int* colid, __global float4* data, __read_only image2d_t vec, __global float* result, int data_align)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (row < row_num)
    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
	    float4 matelem = data[i];
	    float4 vecelem = read_imagef(vec, smp, coord);
            sum = mad(matelem, vecelem, sum);

	    col++;
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    matelem = data[i + data_align];
	    vecelem = read_imagef(vec, smp, coord);
            sum = mad(matelem, vecelem, sum);
	    
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x = sum.x + sum.y + sum.z + sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum.x;
            
    }
}

//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_bcsr_red_28(__global int* rowptr,  __global int* colid, __global float4* data, __global float4* vec, __global float* result, int data_align)
{
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = vec[col];
            sum = mad(matelem, vecelem, sum);
	    matoffset += data_align;
	    matelem = data[matoffset];
            sum2 = mad(matelem, vecelem, sum2);
	    
	    matoffset += data_align;
	    matelem = data[matoffset];
	    vecelem = vec[col + 1];
            sum = mad(matelem, vecelem, sum);
	    matoffset += data_align;
	    matelem = data[matoffset];
            sum2 = mad(matelem, vecelem, sum2);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x = sum.x + sum.y + sum.z + sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2] += sum.x;
        
	s_ptr[localid] = sum2.x = sum2.x + sum2.y + sum2.z + sum2.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 32];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 16];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 8];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 4];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 2];
        sum2.x = sum2.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2 + 1] += sum2.x;
            
            
    }
}


__kernel void gpu_bcsr_red_28_tx(__global int* rowptr,  __global int* colid, __global float4* data, __read_only image2d_t vec, __global float* result, int data_align)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (row < row_num)
    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    int matoffset = i;
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
	    float4 matelem = data[matoffset];
	    float4 vecelem = read_imagef(vec, smp, coord);
            sum = mad(matelem, vecelem, sum);
	    matoffset += data_align;
	    matelem = data[matoffset];
            sum2 = mad(matelem, vecelem, sum2);
	    
	    col++;
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    matoffset += data_align;
	    matelem = data[matoffset];
	    vecelem = read_imagef(vec, smp, coord);
            sum = mad(matelem, vecelem, sum);
	    matoffset += data_align;
	    matelem = data[matoffset];
            sum2 = mad(matelem, vecelem, sum2);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x = sum.x + sum.y + sum.z + sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2] += sum.x;
        
	s_ptr[localid] = sum2.x = sum2.x + sum2.y + sum2.z + sum2.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 32];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 16];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 8];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 4];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 2];
        sum2.x = sum2.x + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2 + 1] += sum2.x;
    }
}

//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_bcsr_red_48(__global int* rowptr,  __global int* colid, __global float4* data, __global float4* vec, __global float4* result, int data_align)
{
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = vec[col];
            float4 tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    
	    matoffset += data_align;
	    matelem = data[matoffset];
	    vecelem = vec[col + 1];
            tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        s_ptr[localid] = sum.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 32];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 16];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 8];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 4];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 2];
        sum.y = sum.y + s_ptr[localid + 1];

        s_ptr[localid] = sum.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 32];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 16];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 8];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 4];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 2];
        sum.z = sum.z + s_ptr[localid + 1];

        s_ptr[localid] = sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 32];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 16];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 8];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 4];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 2];
        sum.w = sum.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum;
            
    }
}


__kernel void gpu_bcsr_red_48_tx(__global int* rowptr,  __global int* colid, __global float4* data, __read_only image2d_t vec, __global float4* result, int data_align)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (row < row_num)
    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = read_imagef(vec, smp, coord);
	    float4 tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    
	    col++;
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    matoffset += data_align;
	    matelem = data[matoffset];
	    vecelem = read_imagef(vec, smp, coord);
	    tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
        }

	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        s_ptr[localid] = sum.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 32];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 16];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 8];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 4];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 2];
        sum.y = sum.y + s_ptr[localid + 1];

        s_ptr[localid] = sum.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 32];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 16];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 8];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 4];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 2];
        sum.z = sum.z + s_ptr[localid + 1];

        s_ptr[localid] = sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 32];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 16];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 8];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 4];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 2];
        sum.w = sum.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum;
            
    }
}

//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_bcsr_red_88(__global int* rowptr,  __global int* colid, __global float4* data, __global float4* vec, __global float4* result, int data_align)
{
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = vec[col];
            float4 tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.x = sum2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.y = sum2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.z = sum2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.w = sum2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    
	    matoffset += data_align;
	    matelem = data[matoffset];
	    vecelem = vec[col + 1];
            tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.x = sum2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.y = sum2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.z = sum2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.w = sum2.w + tmp.x + tmp.y + tmp.z + tmp.w;
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        s_ptr[localid] = sum.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 32];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 16];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 8];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 4];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 2];
        sum.y = sum.y + s_ptr[localid + 1];

        s_ptr[localid] = sum.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 32];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 16];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 8];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 4];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 2];
        sum.z = sum.z + s_ptr[localid + 1];

        s_ptr[localid] = sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 32];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 16];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 8];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 4];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 2];
        sum.w = sum.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2] += sum;
            
        s_ptr[localid] = sum2.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 32];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 16];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 8];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 4];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 2];
        sum2.x = sum2.x + s_ptr[localid + 1];

        s_ptr[localid] = sum2.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 32];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 16];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 8];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 4];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 2];
        sum2.y = sum2.y + s_ptr[localid + 1];

        s_ptr[localid] = sum2.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 32];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 16];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 8];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 4];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 2];
        sum2.z = sum2.z + s_ptr[localid + 1];

        s_ptr[localid] = sum2.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 32];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 16];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 8];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 4];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 2];
        sum2.w = sum2.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2 + 1] += sum2;
            
    }
}


__kernel void gpu_bcsr_red_88_tx(__global int* rowptr,  __global int* colid, __global float4* data, __read_only image2d_t vec, __global float4* result, int data_align)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int localrowptr[2];
    __local float localsum[BCSR_VEC_GROUP_SIZE + WAVEFRONTSIZE/2];
    int row = get_group_id(0);
    int localid = get_local_id(0);

    localsum[localid] = 0.0f;
    if (localid < 2)
    {
        localrowptr[localid] = rowptr[row + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //if (row < row_num)
    {
        int start = localrowptr[0];
        int end = localrowptr[1];
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = start + localid; i < end; i+= BCSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
	    int matoffset = i;
	    float4 matelem = data[matoffset];
	    float4 vecelem = read_imagef(vec, smp, coord);
	    float4 tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.x = sum2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.y = sum2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.z = sum2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.w = sum2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    
	    col++;
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    matoffset += data_align;
	    matelem = data[matoffset];
	    vecelem = read_imagef(vec, smp, coord);
	    tmp = matelem * vecelem;
	    sum.x = sum.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.y = sum.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.z = sum.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum.w = sum.w + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.x = sum2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.y = sum2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.z = sum2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	    matoffset += data_align;
	    matelem = data[matoffset];
            tmp = matelem * vecelem;
	    sum2.w = sum2.w + tmp.x + tmp.y + tmp.z + tmp.w;
        }

	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 32];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 16];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 8];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 4];
        s_ptr[localid] = sum.x = sum.x + s_ptr[localid + 2];
        sum.x = sum.x + s_ptr[localid + 1];

        s_ptr[localid] = sum.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 32];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 16];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 8];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 4];
        s_ptr[localid] = sum.y = sum.y + s_ptr[localid + 2];
        sum.y = sum.y + s_ptr[localid + 1];

        s_ptr[localid] = sum.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 32];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 16];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 8];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 4];
        s_ptr[localid] = sum.z = sum.z + s_ptr[localid + 2];
        sum.z = sum.z + s_ptr[localid + 1];

        s_ptr[localid] = sum.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 32];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 16];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 8];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 4];
        s_ptr[localid] = sum.w = sum.w + s_ptr[localid + 2];
        sum.w = sum.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2] += sum;
            
        s_ptr[localid] = sum2.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 32];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 16];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 8];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 4];
        s_ptr[localid] = sum2.x = sum2.x + s_ptr[localid + 2];
        sum2.x = sum2.x + s_ptr[localid + 1];

        s_ptr[localid] = sum2.y;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 32];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 16];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 8];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 4];
        s_ptr[localid] = sum2.y = sum2.y + s_ptr[localid + 2];
        sum2.y = sum2.y + s_ptr[localid + 1];

        s_ptr[localid] = sum2.z;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 32];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 16];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 8];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 4];
        s_ptr[localid] = sum2.z = sum2.z + s_ptr[localid + 2];
        sum2.z = sum2.z + s_ptr[localid + 1];

        s_ptr[localid] = sum2.w;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localid < 64)
            s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
	s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 32];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 16];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 8];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 4];
        s_ptr[localid] = sum2.w = sum2.w + s_ptr[localid + 2];
        sum2.w = sum2.w + s_ptr[localid + 1];

        if (localid == 0)
            result[row * 2 + 1] += sum2;
            
    }
}

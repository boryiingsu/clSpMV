
#include "constant.h"

__kernel void gpu_bdia(__global int* bandptr, __global int* offset, __global float* data, int aligned_length, int band_num, __global float* vec, __global float* result, int vecoffset)
{
    int row = get_global_id(0);
    __local int l_bandptr[MAX_BAND_NUM + 1];
    __local int l_offset[MAX_BAND_NUM + 1];
    __local float l_vec_band[WORK_GROUP_SIZE + MAX_BAND_WIDTH];
    int local_id = get_local_id(0);
    if (local_id < MAX_BAND_NUM + 1)
    {
	l_bandptr[local_id] = bandptr[local_id];
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float accumulant = result[row];
    for (int i = 0; i < band_num; i++)
    {
	int start = l_bandptr[i];
	int end = l_bandptr[i + 1];
	int diaoffset = l_offset[i];
	int vecid = row + vecoffset + diaoffset;
	int matoffset = row + start * aligned_length;
	int bandwidth = end - start;
	l_vec_band[local_id] = vec[vecid];
	if (local_id < bandwidth)
	{
	    l_vec_band[local_id + WORK_GROUP_SIZE] = vec[vecid + WORK_GROUP_SIZE];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int j = 0; j < bandwidth; j++)
	{
	    float matrixelem = data[matoffset];
	    float vecelem = l_vec_band[local_id + j];
	    accumulant = mad(matrixelem, vecelem, accumulant);
	    matoffset += aligned_length;
	}
    }
    result[row] = accumulant;

}

__kernel void gpu_bdia_nlvec(__global int* bandptr, __global int* offset, __global float* data, int aligned_length, int band_num, __global float* vec, __global float* result, int vecoffset)
{
    int row = get_global_id(0);
    __local int l_bandptr[MAX_BAND_NUM + 1];
    __local int l_offset[MAX_BAND_NUM + 1];
    int local_id = get_local_id(0);
    if (local_id < MAX_BAND_NUM + 1)
    {
	l_bandptr[local_id] = bandptr[local_id];
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float accumulant = result[row];
    for (int i = 0; i < band_num; i++)
    {
	int start = l_bandptr[i];
	int end = l_bandptr[i + 1];
	int diaoffset = l_offset[i];
	int vecid = row + vecoffset + diaoffset;
	int matoffset = row + start * aligned_length;
	int bandwidth = end - start;
	
	for (int j = 0; j < bandwidth; j++)
	{
	    float matrixelem = data[matoffset];
	    float vecelem = vec[vecid + j];
	    accumulant = mad(matrixelem, vecelem, accumulant);
	    matoffset += aligned_length;
	}
    }
    result[row] = accumulant;

}

__kernel void gpu_bdia_g4(__global int* bandptr, __global int* offset, __global float* data, int aligned_length, int band_num, __global float* vec, __global float* result, int vecoffset)
{
    int row = get_global_id(0);
    __local int l_bandptr[MAX_BAND_NUM + 1];
    __local int l_offset[MAX_BAND_NUM + 1];
    __local float l_vec_band[WORK_GROUP_SIZE + MAX_BAND_WIDTH];
    int local_id = get_local_id(0);
    if (local_id < MAX_BAND_NUM + 1)
    {
	l_bandptr[local_id] = bandptr[local_id];
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float accumulant = result[row];
    for (int i = 0; i < band_num; i++)
    {
	int start = l_bandptr[i];
	int end = l_bandptr[i + 1];
	int diaoffset = l_offset[i];
	int vecid = row + vecoffset + diaoffset;
	int matoffset = row + start * aligned_length;
	int bandwidth = end - start;
	l_vec_band[local_id] = vec[vecid];
	if (local_id < bandwidth)
	{
	    l_vec_band[local_id + WORK_GROUP_SIZE] = vec[vecid + WORK_GROUP_SIZE];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int j = 0; j < bandwidth; j++)
	{
	    float matrixelem = data[matoffset];
	    float vecelem = l_vec_band[local_id + j];
	    accumulant = mad(matrixelem, vecelem, accumulant);
	    matoffset += aligned_length;
	}
    }
    result[row] = accumulant;
    row += get_global_size(0);
    accumulant = result[row];
    for (int i = 0; i < band_num; i++)
    {
	int start = l_bandptr[i];
	int end = l_bandptr[i + 1];
	int diaoffset = l_offset[i];
	int vecid = row + vecoffset + diaoffset;
	int matoffset = row + start * aligned_length;
	int bandwidth = end - start;
	l_vec_band[local_id] = vec[vecid];
	if (local_id < bandwidth)
	{
	    l_vec_band[local_id + WORK_GROUP_SIZE] = vec[vecid + WORK_GROUP_SIZE];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int j = 0; j < bandwidth; j++)
	{
	    float matrixelem = data[matoffset];
	    float vecelem = l_vec_band[local_id + j];
	    accumulant = mad(matrixelem, vecelem, accumulant);
	    matoffset += aligned_length;
	}
    }
    result[row] = accumulant;
    row += get_global_size(0);
    accumulant = result[row];
    for (int i = 0; i < band_num; i++)
    {
	int start = l_bandptr[i];
	int end = l_bandptr[i + 1];
	int diaoffset = l_offset[i];
	int vecid = row + vecoffset + diaoffset;
	int matoffset = row + start * aligned_length;
	int bandwidth = end - start;
	l_vec_band[local_id] = vec[vecid];
	if (local_id < bandwidth)
	{
	    l_vec_band[local_id + WORK_GROUP_SIZE] = vec[vecid + WORK_GROUP_SIZE];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int j = 0; j < bandwidth; j++)
	{
	    float matrixelem = data[matoffset];
	    float vecelem = l_vec_band[local_id + j];
	    accumulant = mad(matrixelem, vecelem, accumulant);
	    matoffset += aligned_length;
	}
    }
    result[row] = accumulant;
    row += get_global_size(0);
    accumulant = result[row];
    for (int i = 0; i < band_num; i++)
    {
	int start = l_bandptr[i];
	int end = l_bandptr[i + 1];
	int diaoffset = l_offset[i];
	int vecid = row + vecoffset + diaoffset;
	int matoffset = row + start * aligned_length;
	int bandwidth = end - start;
	l_vec_band[local_id] = vec[vecid];
	if (local_id < bandwidth)
	{
	    l_vec_band[local_id + WORK_GROUP_SIZE] = vec[vecid + WORK_GROUP_SIZE];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int j = 0; j < bandwidth; j++)
	{
	    float matrixelem = data[matoffset];
	    float vecelem = l_vec_band[local_id + j];
	    accumulant = mad(matrixelem, vecelem, accumulant);
	    matoffset += aligned_length;
	}
    }
    result[row] = accumulant;

}

__kernel void gpu_bdia_v4(__global int* bandptr, __global int* offset, __global float4* data, int aligned_length, int band_num, __global float* vec, __global float4* result, int vecoffset)
{
    int row = get_global_id(0);
    __local int l_bandptr[MAX_BAND_NUM + 1];
    __local int l_offset[MAX_BAND_NUM + 1];
    __local float l_vec_band[WORK_GROUP_SIZE * 4 + MAX_BAND_WIDTH];
    int local_id = get_local_id(0);
    int new_row_id = get_group_id(0) * WORK_GROUP_SIZE * 4 + local_id;
    if (local_id < MAX_BAND_NUM + 1)
    {
	l_bandptr[local_id] = bandptr[local_id];
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 accumulant = result[row];
    for (int i = 0; i < band_num; i++)
    {
	int start = l_bandptr[i];
	int end = l_bandptr[i + 1];
	int diaoffset = l_offset[i];
	int matoffset = row + start * aligned_length;
	int bandwidth = end - start;
	int vecid = new_row_id + vecoffset + diaoffset;
	l_vec_band[local_id] = vec[vecid];
	l_vec_band[local_id + WORK_GROUP_SIZE] = vec[vecid + WORK_GROUP_SIZE];
	l_vec_band[local_id + 2 * WORK_GROUP_SIZE] = vec[vecid + 2 * WORK_GROUP_SIZE];
	l_vec_band[local_id + 3 * WORK_GROUP_SIZE] = vec[vecid + 3 * WORK_GROUP_SIZE];
	if (local_id < bandwidth)
	{
	    l_vec_band[local_id + 4 * WORK_GROUP_SIZE] = vec[vecid + 4 * WORK_GROUP_SIZE];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int j = 0; j < bandwidth; j++)
	{
	    float4 matrixelem = data[matoffset];
	    float4 vecelem = vload4(0, l_vec_band + local_id * 4 + j);
	    accumulant = mad(matrixelem, vecelem, accumulant);
	    matoffset += aligned_length;
	}
    }

    result[row] = accumulant;

}

__kernel void gpu_bdia_v4_nlvec(__global int* bandptr, __global int* offset, __global float4* data, int aligned_length, int band_num, __global float* vec, __global float4* result, int vecoffset)
{
    int row = get_global_id(0);
    __local int l_bandptr[MAX_BAND_NUM + 1];
    __local int l_offset[MAX_BAND_NUM + 1];
    int local_id = get_local_id(0);
    if (local_id < MAX_BAND_NUM + 1)
    {
	l_bandptr[local_id] = bandptr[local_id];
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 accumulant = result[row];
    for (int i = 0; i < band_num; i++)
    {
	int start = l_bandptr[i];
	int end = l_bandptr[i + 1];
	int diaoffset = l_offset[i];
	int matoffset = row + start * aligned_length;
	int bandwidth = end - start;
	int vecid = row * 4 + vecoffset + diaoffset;
	
	for (int j = 0; j < bandwidth; j++)
	{
	    float4 matrixelem = data[matoffset];
	    float4 vecelem = vload4(0, vec + vecid + j);
	    accumulant = mad(matrixelem, vecelem, accumulant);
	    matoffset += aligned_length;
	}
    }

    result[row] = accumulant;

}


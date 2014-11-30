#include "constant.h"


__kernel void gpu_ell(__global int* col_id, __global float* data, int aligned_length, int ell_num, __global float* vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    float accumulant = result[row];
    int matoffset = row;
    for (int i = 0; i < ell_num; i++)
    {
	int vecid = col_id[matoffset];
	float matrixelem = data[matoffset];
	float vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += aligned_length;
    }
    
    result[row] = accumulant;
}

__kernel void gpu_ell_v4(__global int4* col_id, __global float4* data, int aligned_length, int ell_num, __global float* vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    float4 accumulant = result[row];
    int matoffset = row;
    for (int i = 0; i < ell_num; i++)
    {
	int4 vecid = col_id[matoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem;
	vecelem.x = vec[vecid.x];
	vecelem.y = vec[vecid.y];
	vecelem.z = vec[vecid.z];
	vecelem.w = vec[vecid.w];
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += aligned_length;
    }
    
    result[row] = accumulant;
}

__kernel void gpu_ell_tx(__global int* col_id, __global float* data, int aligned_length, int ell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float accumulant = result[row];
    int matoffset = row;
    for (int i = 0; i < ell_num; i++)
    {
	int vecid = col_id[matoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem.x, accumulant);
	matoffset += aligned_length;
    }
    result[row] = accumulant;
}

__kernel void gpu_ell_v4_tx(__global int4* col_id, __global float4* data, int aligned_length, int ell_num, __read_only image2d_t vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant = result[row];
    int matoffset = row;
    for (int i = 0; i < ell_num; i++)
    {
	float4 matrixelem = data[matoffset];
	int4 vecid = col_id[matoffset];
	int2 coord;
	coord.x = vecid.x & WIDTH_MASK;
	coord.y = vecid.x >> LOG_WIDTH;
	float4 tmp = read_imagef(vec, smp, coord);
	int2 coord1;
	coord1.x = vecid.y & WIDTH_MASK;
	coord1.y = vecid.y >> LOG_WIDTH;
	float4 tmp1 = read_imagef(vec, smp, coord1);
	int2 coord2;
	coord2.x = vecid.z & WIDTH_MASK;
	coord2.y = vecid.z >> LOG_WIDTH;
	float4 tmp2 = read_imagef(vec, smp, coord2);
	int2 coord3;
	coord3.x = vecid.w & WIDTH_MASK;
	coord3.y = vecid.w >> LOG_WIDTH;
	float4 tmp3 = read_imagef(vec, smp, coord3);
	float4 vecelem;
	vecelem.x = tmp.x;
	vecelem.y = tmp1.x;
	vecelem.z = tmp2.x;
	vecelem.w = tmp3.x;
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += aligned_length;
    }
    result[row] = accumulant;
}


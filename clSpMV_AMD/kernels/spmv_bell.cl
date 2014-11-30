#include "constant.h"


__kernel void gpu_bell14(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    float accumulant = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant = accumulant + tmp.x + tmp.y + tmp.z + tmp.w;
	vecoffset += col_align;
	matoffset += data_align;
    }
    
    result[row] = accumulant;
}

__kernel void gpu_bell14_mad(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant.x = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
	vecoffset += col_align;
	matoffset += data_align;
    }
    
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
}


__kernel void gpu_bell14_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float accumulant = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant = accumulant + tmp.x + tmp.y + tmp.z + tmp.w;
	vecoffset += col_align;
	matoffset += data_align;
    }
    result[row] = accumulant;
}

__kernel void gpu_bell14_mad_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant.x = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem, accumulant);
	vecoffset += col_align;
	matoffset += data_align;
    }
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
}


__kernel void gpu_bell24(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 2;
    float accumulant1 = result[row];
    float accumulant2 = result[row + 1];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant1 = accumulant1 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2 = accumulant2 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant1;
    result[row + 1] = accumulant2;
}

__kernel void gpu_bell24_mad(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 2;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
}


__kernel void gpu_bell24_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 2;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float accumulant1 = result[row];
    float accumulant2 = result[row + 1];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant1 = accumulant1 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2 = accumulant2 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant1;
    result[row + 1] = accumulant2;
}

__kernel void gpu_bell24_mad_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 2;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
}

__kernel void gpu_bell44(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    float4 accumulants = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulants.x = accumulants.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.y = accumulants.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.z = accumulants.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.w = accumulants.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulants;
}

__kernel void gpu_bell44_mad(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 4;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    float4 accumulant3 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant3.x = result[row + 2];
    float4 accumulant4 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant4.x = result[row + 3];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
    result[row + 2] = accumulant3.x + accumulant3.y + accumulant3.z + accumulant3.w;
    result[row + 3] = accumulant4.x + accumulant4.y + accumulant4.z + accumulant4.w;
}


__kernel void gpu_bell44_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulants = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulants.x = accumulants.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.y = accumulants.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.z = accumulants.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.w = accumulants.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulants;
}

__kernel void gpu_bell44_mad_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 4;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    float4 accumulant3 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant3.x = result[row + 2];
    float4 accumulant4 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant4.x = result[row + 3];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
    result[row + 2] = accumulant3.x + accumulant3.y + accumulant3.z + accumulant3.w;
    result[row + 3] = accumulant4.x + accumulant4.y + accumulant4.z + accumulant4.w;
}

__kernel void gpu_bell84(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    int doublerow = row * 2;
    float4 accumulant1 = result[doublerow];
    float4 accumulant2 = result[doublerow + 1];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant1.x = accumulant1.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.y = accumulant1.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.z = accumulant1.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.w = accumulant1.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[doublerow] = accumulant1;
    result[doublerow + 1] = accumulant2;
}

__kernel void gpu_bell84_mad(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 8;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    float4 accumulant3 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant3.x = result[row + 2];
    float4 accumulant4 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant4.x = result[row + 3];
    float4 accumulant5 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant5.x = result[row + 4];
    float4 accumulant6 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant6.x = result[row + 5];
    float4 accumulant7 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant7.x = result[row + 6];
    float4 accumulant8 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant8.x = result[row + 7];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant5 = mad(matrixelem, vecelem, accumulant5);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant6 = mad(matrixelem, vecelem, accumulant6);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant7 = mad(matrixelem, vecelem, accumulant7);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant8 = mad(matrixelem, vecelem, accumulant8);
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
    result[row + 2] = accumulant3.x + accumulant3.y + accumulant3.z + accumulant3.w;
    result[row + 3] = accumulant4.x + accumulant4.y + accumulant4.z + accumulant4.w;
    result[row + 4] = accumulant5.x + accumulant5.y + accumulant5.z + accumulant5.w;
    result[row + 5] = accumulant6.x + accumulant6.y + accumulant6.z + accumulant6.w;
    result[row + 6] = accumulant7.x + accumulant7.y + accumulant7.z + accumulant7.w;
    result[row + 7] = accumulant8.x + accumulant8.y + accumulant8.z + accumulant8.w;
}


__kernel void gpu_bell84_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int doublerow = row * 2;
    float4 accumulant1 = result[doublerow];
    float4 accumulant2 = result[doublerow + 1];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant1.x = accumulant1.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.y = accumulant1.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.z = accumulant1.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.w = accumulant1.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[doublerow] = accumulant1;
    result[doublerow + 1] = accumulant2;
}

__kernel void gpu_bell84_mad_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 8;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    float4 accumulant3 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant3.x = result[row + 2];
    float4 accumulant4 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant4.x = result[row + 3];
    float4 accumulant5 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant5.x = result[row + 4];
    float4 accumulant6 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant6.x = result[row + 5];
    float4 accumulant7 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant7.x = result[row + 6];
    float4 accumulant8 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant8.x = result[row + 7];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant5 = mad(matrixelem, vecelem, accumulant5);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant6 = mad(matrixelem, vecelem, accumulant6);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant7 = mad(matrixelem, vecelem, accumulant7);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant8 = mad(matrixelem, vecelem, accumulant8);
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
    result[row + 2] = accumulant3.x + accumulant3.y + accumulant3.z + accumulant3.w;
    result[row + 3] = accumulant4.x + accumulant4.y + accumulant4.z + accumulant4.w;
    result[row + 4] = accumulant5.x + accumulant5.y + accumulant5.z + accumulant5.w;
    result[row + 5] = accumulant6.x + accumulant6.y + accumulant6.z + accumulant6.w;
    result[row + 6] = accumulant7.x + accumulant7.y + accumulant7.z + accumulant7.w;
    result[row + 7] = accumulant8.x + accumulant8.y + accumulant8.z + accumulant8.w;
}


__kernel void gpu_bell18(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    float accumulant = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant = accumulant + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	vecelem = vec[vecid + 1];
	tmp = matrixelem * vecelem;
	accumulant = accumulant + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant;
}

__kernel void gpu_bell18_mad(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant.x = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += data_align;
	matrixelem = data[matoffset];
	vecelem = vec[vecid + 1];
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
}

__kernel void gpu_bell18_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float accumulant = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant = accumulant + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	tmp = matrixelem * vecelem;
	accumulant = accumulant + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant;
}


__kernel void gpu_bell18_mad_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant.x = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += data_align;
	matrixelem = data[matoffset];
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
}

__kernel void gpu_bell28(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 2;
    float accumulant1 = result[row];
    float accumulant2 = result[row + 1];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant1 = accumulant1 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2 = accumulant2 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;

	vecelem = vec[vecid + 1];
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1 = accumulant1 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2 = accumulant2 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant1;
    result[row + 1] = accumulant2;
}

__kernel void gpu_bell28_mad(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 2;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;

	vecelem = vec[vecid + 1];
	matrixelem = data[matoffset];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
}

__kernel void gpu_bell28_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 2;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float accumulant1 = result[row];
    float accumulant2 = result[row + 1];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant1 = accumulant1 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2 = accumulant2 + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	tmp = matrixelem * vecelem;
	accumulant1 = accumulant1 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2 = accumulant2 + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant1;
    result[row + 1] = accumulant2;
}


__kernel void gpu_bell28_mad_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 2;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
}

__kernel void gpu_bell48(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    float4 accumulants = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulants.x = accumulants.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.y = accumulants.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.z = accumulants.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.w = accumulants.w + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecelem = vec[vecid + 1];
	tmp = matrixelem * vecelem;
	accumulants.x = accumulants.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.y = accumulants.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.z = accumulants.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.w = accumulants.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulants;
}

__kernel void gpu_bell48_mad(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 4;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    float4 accumulant3 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant3.x = result[row + 2];
    float4 accumulant4 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant4.x = result[row + 3];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecelem = vec[vecid + 1];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
    result[row + 2] = accumulant3.x + accumulant3.y + accumulant3.z + accumulant3.w;
    result[row + 3] = accumulant4.x + accumulant4.y + accumulant4.z + accumulant4.w;
}

__kernel void gpu_bell48_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulants = result[row];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulants.x = accumulants.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.y = accumulants.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.z = accumulants.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.w = accumulants.w + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	tmp = matrixelem * vecelem;
	accumulants.x = accumulants.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.y = accumulants.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.z = accumulants.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulants.w = accumulants.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulants;
}


__kernel void gpu_bell48_mad_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 4;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    float4 accumulant3 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant3.x = result[row + 2];
    float4 accumulant4 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant4.x = result[row + 3];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
    result[row + 2] = accumulant3.x + accumulant3.y + accumulant3.z + accumulant3.w;
    result[row + 3] = accumulant4.x + accumulant4.y + accumulant4.z + accumulant4.w;
}

__kernel void gpu_bell88(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    int doublerow = row * 2;
    float4 accumulant1 = result[doublerow];
    float4 accumulant2 = result[doublerow + 1];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant1.x = accumulant1.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.y = accumulant1.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.z = accumulant1.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.w = accumulant1.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecelem = vec[vecid + 1];
	tmp = matrixelem * vecelem;
	accumulant1.x = accumulant1.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.y = accumulant1.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.z = accumulant1.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.w = accumulant1.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[doublerow] = accumulant1;
    result[doublerow + 1] = accumulant2;
}

__kernel void gpu_bell88_mad(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __global float4* vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 8;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    float4 accumulant3 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant3.x = result[row + 2];
    float4 accumulant4 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant4.x = result[row + 3];
    float4 accumulant5 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant5.x = result[row + 4];
    float4 accumulant6 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant6.x = result[row + 5];
    float4 accumulant7 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant7.x = result[row + 6];
    float4 accumulant8 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant8.x = result[row + 7];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant5 = mad(matrixelem, vecelem, accumulant5);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant6 = mad(matrixelem, vecelem, accumulant6);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant7 = mad(matrixelem, vecelem, accumulant7);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant8 = mad(matrixelem, vecelem, accumulant8);

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecelem = vec[vecid + 1];
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant5 = mad(matrixelem, vecelem, accumulant5);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant6 = mad(matrixelem, vecelem, accumulant6);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant7 = mad(matrixelem, vecelem, accumulant7);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant8 = mad(matrixelem, vecelem, accumulant8);
	matoffset += data_align;
	vecoffset += col_align;
    }
    
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
    result[row + 2] = accumulant3.x + accumulant3.y + accumulant3.z + accumulant3.w;
    result[row + 3] = accumulant4.x + accumulant4.y + accumulant4.z + accumulant4.w;
    result[row + 4] = accumulant5.x + accumulant5.y + accumulant5.z + accumulant5.w;
    result[row + 5] = accumulant6.x + accumulant6.y + accumulant6.z + accumulant6.w;
    result[row + 6] = accumulant7.x + accumulant7.y + accumulant7.z + accumulant7.w;
    result[row + 7] = accumulant8.x + accumulant8.y + accumulant8.z + accumulant8.w;
}

__kernel void gpu_bell88_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float4* result, int row_num)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int doublerow = row * 2;
    float4 accumulant1 = result[doublerow];
    float4 accumulant2 = result[doublerow + 1];
    int vecoffset = row;
    int matoffset = row;
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant1.x = accumulant1.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.y = accumulant1.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.z = accumulant1.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.w = accumulant1.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	tmp = matrixelem * vecelem;
	accumulant1.x = accumulant1.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.y = accumulant1.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.z = accumulant1.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant1.w = accumulant1.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[doublerow] = accumulant1;
    result[doublerow + 1] = accumulant2;
}


__kernel void gpu_bell88_mad_tx(__global int* col_id, __global float4* data, int data_align, int col_align, int bell_num, __read_only image2d_t vec, __global float* result, int row_num)
{
    int row = get_global_id(0) * 8;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    float4 accumulant1 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant1.x = result[row];
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant2.x = result[row + 1];
    float4 accumulant3 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant3.x = result[row + 2];
    float4 accumulant4 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant4.x = result[row + 3];
    float4 accumulant5 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant5.x = result[row + 4];
    float4 accumulant6 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant6.x = result[row + 5];
    float4 accumulant7 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant7.x = result[row + 6];
    float4 accumulant8 = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant8.x = result[row + 7];
    int vecoffset = get_global_id(0);
    int matoffset = get_global_id(0);
    for (int i = 0; i < bell_num; i++)
    {
	int vecid = col_id[vecoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant5 = mad(matrixelem, vecelem, accumulant5);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant6 = mad(matrixelem, vecelem, accumulant6);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant7 = mad(matrixelem, vecelem, accumulant7);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant8 = mad(matrixelem, vecelem, accumulant8);

	matoffset += data_align;
	matrixelem = data[matoffset];
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	accumulant1 = mad(matrixelem, vecelem, accumulant1);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant3 = mad(matrixelem, vecelem, accumulant3);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant4 = mad(matrixelem, vecelem, accumulant4);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant5 = mad(matrixelem, vecelem, accumulant5);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant6 = mad(matrixelem, vecelem, accumulant6);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant7 = mad(matrixelem, vecelem, accumulant7);
	matoffset += data_align;
	matrixelem = data[matoffset];
	accumulant8 = mad(matrixelem, vecelem, accumulant8);
	matoffset += data_align;
	vecoffset += col_align;
    }
    result[row] = accumulant1.x + accumulant1.y + accumulant1.z + accumulant1.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
    result[row + 2] = accumulant3.x + accumulant3.y + accumulant3.z + accumulant3.w;
    result[row + 3] = accumulant4.x + accumulant4.y + accumulant4.z + accumulant4.w;
    result[row + 4] = accumulant5.x + accumulant5.y + accumulant5.z + accumulant5.w;
    result[row + 5] = accumulant6.x + accumulant6.y + accumulant6.z + accumulant6.w;
    result[row + 6] = accumulant7.x + accumulant7.y + accumulant7.z + accumulant7.w;
    result[row + 7] = accumulant8.x + accumulant8.y + accumulant8.z + accumulant8.w;
}


#include "constant.h"


__kernel void gpu_csr_ve_slm_pm_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num)
{
    __local int localrowptr[CSR_VEC_GROUP_SIZE / WARPSIZE][2];
    __local float localsum[CSR_VEC_GROUP_SIZE + WARPSIZE / 2];
    int row = get_global_id(0) / WARPSIZE;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    int warpnum = get_global_size(0) / WARPSIZE;
    //int row = get_group_id(0) + get_num_groups(0) * warpid;

    for (; row < row_num; row += warpnum)
    {
	
        if (laneid < 2)
        {
            localrowptr[warpid][laneid] = rowptr[row + laneid];
        }
	
        int start = localrowptr[warpid][0];
        int end = localrowptr[warpid][1];
        float sum = 0.0f;
        for (int i = start + laneid; i < end; i+= WARPSIZE)
        {
            int col = colid[i];
            sum = mad(data[i], vec[col], sum);
        }
	
	volatile __local float* s_ptr = localsum;
	s_ptr[localid] = sum;
        s_ptr[localid] = sum = sum + s_ptr[localid + 16];
        s_ptr[localid] = sum = sum + s_ptr[localid + 8];
        s_ptr[localid] = sum = sum + s_ptr[localid + 4];
        s_ptr[localid] = sum = sum + s_ptr[localid + 2];
        sum = sum + s_ptr[localid + 1];
	
        if (laneid == 0)
            result[row] += sum;
            
    }
}



__kernel void gpu_csr_ve_reduction_fs(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num)
{
    __local float localsum[CSR_VEC_GROUP_SIZE + WAVEFRONTSIZE / 2];
    __local int localrowptr[2];
    int row = get_group_id(0);
    int localid = get_local_id(0);
    int groupsize = get_num_groups(0);

    localsum[localid] = 0.0f;

    for (; row < row_num; row += groupsize)
    {
	if (localid < 2)
	{
	    localrowptr[localid] = rowptr[row + localid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

        int start = localrowptr[0];
        int end = localrowptr[1];
        float sum = 0.0f;
        for (int i = start + localid; i < end; i+= CSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
            sum = mad(data[i], vec[col], sum);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum;
	s_ptr[localid] = sum = sum + s_ptr[localid + 32];
        s_ptr[localid] = sum = sum + s_ptr[localid + 16];
        s_ptr[localid] = sum = sum + s_ptr[localid + 8];
        s_ptr[localid] = sum = sum + s_ptr[localid + 4];
        s_ptr[localid] = sum = sum + s_ptr[localid + 2];
        sum = sum + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum;
            
    }
}


//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows fixed global num
__kernel void gpu_csr_ve_slm_pm_fs_tx(__global int* rowptr,  __global int* colid, __global float* data, __read_only image2d_t vec, __global float* result, int row_num)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local float localsum[CSR_VEC_GROUP_SIZE + WARPSIZE / 2];
    __local int localrowptr[CSR_VEC_GROUP_SIZE / WARPSIZE][2];
    int row = get_global_id(0) / WARPSIZE;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    //int row = get_group_id(0) + get_num_groups(0) * warpid;
    int warpnum = get_global_size(0) / WARPSIZE;


    for (; row < row_num; row += warpnum)
    {
        if (laneid < 2)
        {
            localrowptr[warpid][laneid] = rowptr[row + laneid];
        }

        int start = localrowptr[warpid][0];
        int end = localrowptr[warpid][1];
        float sum = 0.0f;
        for (int i = start + laneid; i < end; i+= WARPSIZE)
        {
            int col = colid[i];
	    int2 coord;
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    float4 ans = read_imagef(vec, smp, coord);
            sum = mad(data[i], ans.x, sum);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum;
        s_ptr[localid] = sum = sum + s_ptr[localid + 16];
        s_ptr[localid] = sum = sum + s_ptr[localid + 8];
        s_ptr[localid] = sum = sum + s_ptr[localid + 4];
        s_ptr[localid] = sum = sum + s_ptr[localid + 2];
        sum = sum + s_ptr[localid + 1];

        if (laneid == 0)
            result[row] += sum;
            
    }
}



//Use a warp for a row. Warp size = 16 slm rowptr padd matrix strided rows mad float4
__kernel void gpu_csr_ve_reduction_fs_tx(__global int* rowptr,  __global int* colid, __global float* data, __read_only image2d_t vec, __global float* result, int row_num)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local float localsum[CSR_VEC_GROUP_SIZE + WAVEFRONTSIZE / 2];
    __local int localrowptr[2];
    int row = get_group_id(0);
    int localid = get_local_id(0);
    int groupsize = get_num_groups(0);

    localsum[localid] = 0.0f;

    for (; row < row_num; row += groupsize)
    {
	if (localid < 2)
	{
	    localrowptr[localid] = rowptr[row + localid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

        int start = localrowptr[0];
        int end = localrowptr[1];
        float sum = 0.0f;
        for (int i = start + localid; i < end; i+= CSR_VEC_GROUP_SIZE)
        {
            int col = colid[i];
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
		float4 ans = read_imagef(vec, smp, coord);
            sum = mad(data[i], ans.x, sum);
        }
	volatile __local float* s_ptr = localsum;
        s_ptr[localid] = sum;
	s_ptr[localid] = sum = sum + s_ptr[localid + 32];
        s_ptr[localid] = sum = sum + s_ptr[localid + 16];
        s_ptr[localid] = sum = sum + s_ptr[localid + 8];
        s_ptr[localid] = sum = sum + s_ptr[localid + 4];
        s_ptr[localid] = sum = sum + s_ptr[localid + 2];
        sum = sum + s_ptr[localid + 1];

        if (localid == 0)
            result[row] += sum;
            
    }
}


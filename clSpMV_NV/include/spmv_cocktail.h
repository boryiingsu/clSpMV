#ifndef __SPMV_ULHY__H__
#define __SPMV_ULHY__H__

#include "CL/cl.h"
#include "analyze.h"

struct cocktail_gpu
{
    //Shared
    cl_mem vec;
    cl_mem vec_tex_r;
    cl_mem vec_tex_rgba;
    cl_mem vec_image;
    cl_mem res;

    //BDIA and DIA Shared
    cl_mem vec_extended;
    int    vec_offset;

    //BDIA
    cl_mem bdia_band_ptr;
    cl_mem bdia_offsets;
    cl_mem bdia_data;
    int	   bdia_length;
    int    bdia_length4;
    int    bdia_bandnum;

    //DIA
    cl_mem dia_offsets;
    cl_mem dia_data;
    int    dia_length;
    int    dia_length4;
    int    dia_dianum;

    //SBELL
    cl_mem sbell_slice_ptr;
    cl_mem sbell_col_id;
    cl_mem sbell_data;
    int    sbell_slicenum;

    //BELL
    cl_mem bell_col_id;
    cl_mem bell_data;
    int    bell_data_align;
    int    bell_col_align;
    int    bell_ellnum;
    int    bell_brownum;

    //BCSR
    cl_mem bcsr_row_ptr;
    cl_mem bcsr_col_id;
    cl_mem bcsr_data;
    int    bcsr_data_align;

    //SELL
    cl_mem sell_slice_ptr;
    cl_mem sell_col_id;
    cl_mem sell_data;
    int    sell_slicenum;

    //ELL
    cl_mem ell_col_id;
    cl_mem ell_data;
    int    ell_length;
    int    ell_length4;
    int    ell_ellnum;
    int    ell_rownum;
    int    ell_rownum4;

    //CSR
    cl_mem csr_row_ptr;
    cl_mem csr_col_id;
    cl_mem csr_data;
    int    csr_rownum;

    //COO
    cl_mem coo_row_id;
    cl_mem coo_col_id;
    cl_mem coo_data;
    cl_mem coo_tmp_row;
    cl_mem coo_tmp_data;
    int    coo_process_size;
    int    coo_paddednnz;
    int    coo_activewarp;
    int    coo_group_num;

    //Flags
    bool ifusebdia;
    bool ifusedia;
    bool ifusesbell;
    bool ifusebell;
    bool ifusebcsr;
    bool ifusesell;
    bool ifuseell;
    bool ifusecsr;
    bool ifusecoo;

    //Flags of using texture memory to store vector
    bool ifteximage;
    bool iftexr;
    bool iftexrgba;
    bool ifsbelltex;
    bool ifbelltex;
    bool ifbcsrtex;
    bool ifselltex;
    bool ifelltex;
    bool ifcsrtex;
    bool ifcootex;
};

void init_cocktail_gpu(cocktail_gpu& mat);
void free_cocktail_gpu(cocktail_gpu& mat);

struct cocktail_kernels
{
    //OpenCL related environment variables
    cl_device_id* devices;
    cl_context context;
    cl_command_queue cmdQueue;
    cl_program program;

    //Kernels
    cl_kernel bdia_kernel;
    size_t bdia_block[2];
    size_t bdia_global[2];

    cl_kernel dia_kernel;
    size_t dia_block[2];
    size_t dia_global[2];

    cl_kernel sbell_kernel;
    size_t sbell_block[2];
    size_t sbell_global[2];

    cl_kernel bell_kernel;
    size_t bell_block[2];
    size_t bell_global[2];

    cl_kernel bcsr_kernel;
    size_t bcsr_block[2];
    size_t bcsr_global[2];

    cl_kernel sell_kernel;
    size_t sell_block[2];
    size_t sell_global[2];

    cl_kernel ell_kernel;
    size_t ell_block[2];
    size_t ell_global[2];

    cl_kernel csr_kernel;
    size_t csr_block[2];
    size_t csr_global[2];

    cl_kernel coo_kernel_s1;
    size_t coo_block_s1[2];
    size_t coo_global_s1[2];
    cl_kernel coo_kernel_s2;
    size_t coo_block_s2[2];
    size_t coo_global_s2[2];
};

void init_cocktail_kernels(cocktail_kernels& cocktail);
void free_cocktail_kernels(cocktail_kernels& cocktail);

void evaluate(cocktail<int, int, float>& cpumat, float* vec, float* res, float* coores);
void cpy_vector_from_cpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* vec, int vec_size);
void cpy_vector_from_gpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, cl_mem vec, int vec_size);
void cpy_result_from_cpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* res, int res_size);
void cpy_result_from_gpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, cl_mem res, int res_size);
void cpy_result_to_cpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* res, int res_size);
void cpy_result_to_gpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, cl_mem res, int res_size);
void init_mat_kernels(cocktail<int, int, float>& cpumat, float* vec, float* res, cocktail_gpu& gpumat, cocktail_kernels& kernels, bool use_tex);
void free_mat_kernels(cocktail_gpu& gpumat, cocktail_kernels& kernels);
void do_spmv(cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, int ntimes);



#endif


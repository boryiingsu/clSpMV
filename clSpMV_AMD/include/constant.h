#ifndef __CONSTANT_H__
#define __CONSTANT_H__

//For creating opencl gpu programs
#define CONTEXTTYPE CL_DEVICE_TYPE_GPU

//For matrix_storage.h
#define DIAINCSIZE 200
#define MAX_LEVELS  1000
#define MAX_MEM_OBJ 268435456

//For texture mapping 1d to 2d
#define WIDTH_MASK 1023
#define LOG_WIDTH 10
#define VEC2DWIDTH 1024

//For alignment purpose
#define GPU_ALIGNMENT 32
#define COO_ALIGNMENT 32

//Warp info
#define WARPSIZE 32
#define MAX_WARP_NUM 480 // 32 (warps per processor) * 15 processors
#define MAX_WARP_PER_PROC 48 //The maximum number of warps in a processor

//AMD Wavefront Info
#define WAVEFRONTSIZE 64
#define MIN_WAVE_PER_GROUP 4

//Work group size
#define WORK_GROUP_SIZE 256

//Format specific parameters
#define MAX_BAND_NUM 31
#define MAX_BAND_WIDTH 128
#define MAX_DIA_NUM 128
#define CSR_VEC_GROUP_SIZE 64
#define BCSR_VEC_GROUP_SIZE 128
#define CSR_VEC_MIN_TH_NUM 5760 
#define BELL_GROUP_SIZE 256
#define SELL_GROUP_SIZE 128
#define COO_GROUP_SIZE 64

//For benchmarking parameters
#define BDIA_IMP_NUM 5
#define DIA_IMP_NUM 4
#define SBELL_IMP_NUM 1
#define BELL_IMP_NUM 2
#define BCSR_IMP_NUM 1
#define SELL_IMP_NUM 2
#define ELL_IMP_NUM 2
#define CSR_IMP_NUM 2
#define COO_IMP_NUM 1

#endif

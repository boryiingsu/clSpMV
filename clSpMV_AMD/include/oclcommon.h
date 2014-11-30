#ifndef OCLCOMMON__H__
#define OCLCOMMON__H__


#include <stdlib.h>
#include <string.h>

#include "CL/cl.h"

#define ASSERT(x) {if(x == false) {fprintf(stderr, "Error at line %d\n", __LINE__); exit(1);}}
#define CHECKERROR {if (errorCode != CL_SUCCESS) {fprintf(stderr, "Error at line %d code %d message %s\n", __LINE__, errorCode, print_cl_errstring(errorCode)); exit(1);}}

#define ALLOCATE_GPU_READ(deviceBuf, HostBuf, mem_size) \
    if(errorCode == CL_SUCCESS) { \
    deviceBuf = clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, mem_size, HostBuf, &errorCode); \
    clEnqueueWriteBuffer(cmdQueue, deviceBuf, CL_TRUE, 0, mem_size, HostBuf, 0, NULL, NULL); \
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } \
    }

#define ALLOCATE_GPU_READ_WRITE_INIT(deviceBuf, HostBuf, mem_size) \
    if(errorCode == CL_SUCCESS) { \
    deviceBuf = clCreateBuffer(context,  CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, mem_size, HostBuf, &errorCode); \
    clEnqueueWriteBuffer(cmdQueue, deviceBuf, CL_TRUE, 0, mem_size, HostBuf, 0, NULL, NULL); \
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } \
    }

#define ALLOCATE_GPU_READ_IMAGE(deviceImage, HostBuf, num_elements) \
    deviceImage = clCreateImage2D(context, CL_MEM_READ_ONLY, &imageFormat, num_elements, 1, 0, NULL, &errorCode); \
    if(errorCode != CL_SUCCESS ) {printf("Error: clCreateImage2D failed on %s\n", ##deviceImage); }\
    const size_t region##deviceImage[] = { num_elements, 1, 1,}; \
    errorCode = clEnqueueWriteImage(cmdQueue, deviceImage, CL_TRUE, origin, region##deviceImage, 0, 0, HostBuf, 0, NULL, NULL); \
    if(errorCode != CL_SUCCESS ) {printf("Error: clEnqueueWriteImage failed on %s\n", ##deviceImage); }\

#define ALLOCATE_GPU_WRITE(deviceBuf, HostBuf, mem_size) \
    if(errorCode == CL_SUCCESS) { \
    deviceBuf = clCreateBuffer(context,  CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, mem_size, HostBuf, &errorCode); \
    if( errorCode != CL_SUCCESS ) { printf("Error: clCreateBuffer returned %d\n", errorCode); } \
    }


double executionTime(cl_event &event);
bool LoadSourceFromFile(const char* filename, char* & sourceCode );
char *print_cl_errstring(cl_int err);
cl_int PrintDeviceInfo(cl_device_id* devices, size_t numDevices );
int initialization(cl_device_type deviceType, cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program, char* clFileName);
void freeObjects(cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program);
void writeProgramBinary(cl_program& program, char* binname, int deviceid);

#endif

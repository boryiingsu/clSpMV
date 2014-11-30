#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include "oclcommon.h"

double executionTime(cl_event &event)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    return (double)(end - start) * 1e-9f;
}


bool LoadSourceFromFile(
    const char* filename,
    char* & sourceCode )
{
    bool error = false;
    FILE* fp = NULL;
    int nsize = 0;

    // Open the shader file

    fp = fopen(filename, "rb");
    if( !fp )
    {
        error = true;
    }
    else
    {
        // Allocate a buffer for the file contents
        fseek( fp, 0, SEEK_END );
        nsize = ftell( fp );
        fseek( fp, 0, SEEK_SET );

        sourceCode = new char [ nsize + 1 ];
        if( sourceCode )
        {
            fread( sourceCode, 1, nsize, fp );
            sourceCode[ nsize ] = 0; // Don't forget the NULL terminator
        }
        else
        {
            error = true;
        }

        fclose( fp );
    }

    return error;
}


char *print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
} 

cl_int PrintDeviceInfo(
    cl_device_id* devices,
    size_t numDevices )
{
    cl_int  errorCode = CL_SUCCESS;

    cl_device_type  deviceType;
    cl_uint         deviceVendorId;
    cl_uint         deviceMaxComputeUnits;
    cl_uint         deviceMaxWorkItemDimensions;
    size_t          deviceMaxWorkItemSizes[16];
    size_t          deviceMaxWorkGroupSize;
    cl_uint         deviceMaxClockFrequency;
    cl_bool         deviceImageSupport;
    size_t	    deviceImageMaxWidth;
    size_t	    deviceImageMaxHeight;
    cl_ulong	    deviceMaxMemAlloc;

    int i = 0;
    for( i = 0; i < numDevices; i++ )
    {
	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_TYPE,
		sizeof( deviceType ),
		&deviceType,
		NULL );
	printf("Errorcode: %d\n", errorCode);
	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_VENDOR_ID,
		sizeof( deviceVendorId ),
		&deviceVendorId,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_MAX_COMPUTE_UNITS,
		sizeof( deviceMaxComputeUnits ),
		&deviceMaxComputeUnits,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
		sizeof( deviceMaxWorkItemDimensions ),
		&deviceMaxWorkItemDimensions,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_MAX_WORK_ITEM_SIZES,
		sizeof( deviceMaxWorkItemSizes ),
		&deviceMaxWorkItemSizes,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof( deviceMaxWorkGroupSize ),
		&deviceMaxWorkGroupSize,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_MAX_CLOCK_FREQUENCY,
		sizeof( deviceMaxClockFrequency ),
		&deviceMaxClockFrequency,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_IMAGE_SUPPORT,
		sizeof( deviceImageSupport ),
		&deviceImageSupport,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_IMAGE2D_MAX_WIDTH,
		sizeof( deviceImageMaxWidth ),
		&deviceImageMaxWidth,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_IMAGE2D_MAX_HEIGHT,
		sizeof( deviceImageMaxHeight ),
		&deviceImageMaxHeight,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	errorCode |= clGetDeviceInfo(
		devices[i],
		CL_DEVICE_MAX_MEM_ALLOC_SIZE,
		sizeof( deviceMaxMemAlloc ),
		&deviceMaxMemAlloc,
		NULL );
	printf("Errorcode: %d\n", errorCode);

	if( 1/*errorCode == CL_SUCCESS*/ )
	{
	    printf("Device[%d]:\n", i );

	    switch( deviceType )
	    {
		case CL_DEVICE_TYPE_DEFAULT:    printf("\tType: \t%s\n", "DEFAULT" );       break;
		case CL_DEVICE_TYPE_CPU:        printf("\tType: \t%s\n", "CPU" );           break;
		case CL_DEVICE_TYPE_GPU:        printf("\tType: \t%s\n", "GPU" );           break;
		case CL_DEVICE_TYPE_ACCELERATOR:printf("\tType: \t%s\n", "ACCELERATOR" );   break;
		default:                        printf("\tType: \t%s\n", "***UNKNOWN***" ); break;
	    }

	    printf("\tVendor Id: \t%x\n", deviceVendorId );
	    printf("\tMax Compute Units: \t%d\n", deviceMaxComputeUnits );
	    printf("\tMax Work Item Dimensions: \t%d\n", deviceMaxWorkItemDimensions );

	    int   j = 0;
	    for( j = 0; j < deviceMaxWorkItemDimensions; j++ )
	    {
		printf("\tMax Work Item Sizes[%d]: \t%d\n", j, deviceMaxWorkItemSizes[j] );
	    }

	    printf("\tMax Work Group Size: \t%d\n", deviceMaxWorkGroupSize );
	    printf("\tMax Clock Frequency: \t%d\n", deviceMaxClockFrequency );
	    printf("\tImage Support?  %s\n", deviceImageSupport ? "true" : "false" );
	    printf("\tImage Max 2D Width  \t%d\n", deviceImageMaxWidth);
	    printf("\tImage Max 2D Height \t%d\n", deviceImageMaxHeight );
	    printf("\tMax Mem Alloc Size \t%d \t%d MB\n", deviceMaxMemAlloc, deviceMaxMemAlloc/(1024*1024) );
	    printf("---\n");
	}
	else
	{
	    printf("Error getting device info for device %d.\n", i);
	}
    }

    return errorCode;
}

int initialization(cl_device_type deviceType, cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program, char* clFileName)
{
    size_t devicesSize = 0;
    size_t numDevices = 0;

    cl_int errorCode = CL_SUCCESS;

    if (devices != NULL)
    {
	printf("Device id not NULL\n");	
	return -1;
    }
    if ((*context) != NULL)
    {
	printf("Context not NULL\n");	
	return -1;
    }
    if ((*cmdQueue) != NULL)
    {
	printf("Command Queue not NULL\n");	
	return -1;
    }
    if ((*program) != NULL)
    {
	printf("Program not NULL\n");	
	return -1;
    }

    char* programSource = NULL;

    if( errorCode == CL_SUCCESS )
    {
	printf("Program File Name: %s\n", clFileName );
	printf("---\n");
    }

    if( errorCode == CL_SUCCESS )
    {
	// Load the kernel source from the passed in file.
	if( LoadSourceFromFile( clFileName, programSource ) )
	{
	    printf("Error: Couldn't load kernel source from file '%s'.\n", clFileName );
	    errorCode = CL_INVALID_OPERATION;
	}
    }


    if( errorCode == CL_SUCCESS )
    {
	cl_uint size_ret = 0;
	errorCode = clGetPlatformIDs(0, NULL, &size_ret);
	if ( (errorCode != CL_SUCCESS) || (size_ret == 0) )
	{
	    return -1;
	}

	cl_platform_id * platforms = new cl_platform_id[size_ret];

	errorCode = clGetPlatformIDs(size_ret, platforms, NULL);

	if ( errorCode != CL_SUCCESS )
	{
	    delete[] platforms;
	    return -1;
	}

	cl_context_properties  properties[3] =  {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], NULL };

	// Create OpenCL device and context.
	(*context) = clCreateContextFromType(properties, deviceType, NULL, NULL, &errorCode);
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clCreateContextFromType() returned %d.\n", errorCode);
	}
	delete[] platforms;
    }

    if( errorCode == CL_SUCCESS )
    {
	// Get the number of devices associated with the context.
	errorCode = clGetContextInfo(*context, CL_CONTEXT_DEVICES, 0, NULL, &devicesSize );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clGetContextInfo() to get num devices returned %d.\n", errorCode);
	}
    }

    if( errorCode == CL_SUCCESS )
    {
	// Get the list of devices associated with the context.
	numDevices = devicesSize / sizeof( cl_device_id );
	devices = new cl_device_id [ numDevices ];

	errorCode = clGetContextInfo(*context, CL_CONTEXT_DEVICES, devicesSize, devices, NULL );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clGetContextInfo() to get list of devices returned %d.\n", errorCode);
	}
    }

    if( errorCode == CL_SUCCESS )
    {
	// Print the device info.
	//errorCode = PrintDeviceInfo(devices, numDevices);
    }

    if( errorCode == CL_SUCCESS )
    {
	// Create a command queue.
	(*cmdQueue) = clCreateCommandQueue(*context, devices[ 0 ], CL_QUEUE_PROFILING_ENABLE, &errorCode );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clCreateCommandQueue() returned %d.\n", errorCode);
	}
    }

    if( errorCode == CL_SUCCESS )
    {
	// Create program.
	(*program) = clCreateProgramWithSource(*context, 1, ( const char** )&programSource, NULL, &errorCode );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clCreateProgramWithSource() returned %d.\n", errorCode );
	}

	// No longer need the programSource char array
	delete [] programSource;
	programSource = NULL;
    }


    if( errorCode == CL_SUCCESS )
    {
	// Build the program.
	errorCode = clBuildProgram(*program, 1, &devices[ 0 ], "", NULL, NULL );
	if( errorCode != CL_SUCCESS )
	{
	    printf("Error: clBuildProgram() returned %d.\n", errorCode );
	}
	//if( errorCode != CL_SUCCESS )
	{

	    size_t  buildLogSize = 0;
	    clGetProgramBuildInfo(*program, devices[ 0 ], CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize );
	    cl_char*    buildLog = new cl_char[ buildLogSize ];
	    if( buildLog )
	    {
		clGetProgramBuildInfo(*program, devices[ 0 ], CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, NULL );

		printf(">>> Build Log:\n");
		printf("%s\n", buildLog );
		printf("<<< End of Build Log\n");
	    }
	}
    }

    if (programSource)
    {
	delete[] programSource;
	programSource = NULL;
    }

    return 1;
}

void freeObjects(cl_device_id* devices, cl_context* context, cl_command_queue* cmdQueue, cl_program* program)
{
    if (*program)
	clReleaseProgram(*program);
    if (*cmdQueue)
	clReleaseCommandQueue(*cmdQueue);
    if (devices)
    {
	delete[] devices;
	devices = NULL;
    }
    if (*context)
	clReleaseContext(*context);
}

void writeProgramBinary(cl_program& program, char* binname, int deviceid)
{
    using namespace std;
    ofstream outfile;
    outfile.open(binname);
    cl_uint numDevices = 0;
    cl_int errorCode = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL); CHECKERROR;
    cout<<"Number of Devices: "<<numDevices<<endl;

    size_t binary_sizes[numDevices];
    errorCode = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*numDevices, binary_sizes, NULL); CHECKERROR;

    char **binaries = new char*[numDevices];
    for (size_t i = 0; i < numDevices; i++)
	binaries[i] = new char[binary_sizes[i] + 1];
    errorCode = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(size_t)*numDevices, binaries, NULL); CHECKERROR;

    binaries[deviceid][binary_sizes[deviceid]] = '\0';
    cout << "Program " << deviceid << ":" << endl;
    outfile << binaries[deviceid];
    outfile.close();

    for (size_t i = 0; i < numDevices; i++)
	delete [] binaries[i];

    delete [] binaries;

}




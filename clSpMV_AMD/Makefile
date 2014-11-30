################################################################################
#
# Common build script
#
################################################################################

.SUFFIXES : .cl

# Basic directory setup for SDK
# (override directories only if they are not already defined)

OCLROOTDIR := /opt/AMDAPP
OCLCOMMONDIR ?= $(OCLROOTDIR)
OCLBINDIR ?= $(OCLROOTDIR)/bin/
OCLLIBDIR     := $(OCLCOMMONDIR)/lib/x86_64

SRCDIR     ?= ./src
ROOTDIR    ?= .
ROOTOBJDIR ?= obj
BINDIR     ?= $(ROOTDIR)/linux
INCDIR	?= $(ROOTDIR)/include


# Add source files here
# C/C++ source files (compiled with gcc / c++)
COMMONFILES		:= util.cpp fileio.cpp oclcommon.cpp
SINGLEFILES		:= mem_bandwidth.cpp spmv_csr_scalar.cpp spmv_csr_vector.cpp spmv_bdia.cpp spmv_dia.cpp spmv_ell.cpp spmv_coo.cpp spmv_bell.cpp spmv_bcsr.cpp spmv_sell.cpp spmv_sbell.cpp spmv_all.cpp
BENCHFILES    		:= bench_bdia.cpp bench_dia.cpp bench_bell.cpp bench_sbell.cpp bench_bcsr.cpp bench_sell.cpp bench_ell.cpp bench_csr.cpp bench_coo.cpp bench_overhead.cpp
COCKTAILFILES    	:= analyze.cpp spmv_cocktail.cpp eval.cpp
################################################################################


# detect if 32 bit or 64 bit system
HP_64 =	$(shell uname -m | grep 64)
OSARCH= $(shell uname -m)


# Compilers
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I$(INCDIR) -I$(OCLCOMMONDIR)/include

ifeq "$(strip $(HP_64))" ""
	MACHINE := 32
	USRLIBDIR := -L/usr/lib/
else
	MACHINE := 64
	USRLIBDIR := -L/usr/lib64/
endif


# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \


# architecture flag for nvcc and gcc compilers build
LIB_ARCH        := $(OSARCH)


# Compiler-specific flags
CXXFLAGS  := $(CXXWARN_FLAGS) 
CFLAGS    := $(CWARN_FLAGS) 

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O3 
	BINSUBDIR   := release
	LIBSUFFIX   :=
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif


# Libs
LIB       := ${USRLIBDIR} -L${OCLLIBDIR}
LIB += -lOpenCL ${OPENGLLIB} ${LIB} 


# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	TARGETDIR := $(OCLLIBDIR)
	TARGET   := $(subst .a,_$(LIB_ARCH)$(LIBSUFFIX).a,$(OCLLIBDIR)/$(STATIC_LIB))
	LINKLINE  = ar qv $(TARGET) $(OBJS) 
else
	TARGETDIR := $(BINDIR)/$(BINSUBDIR)
	TARGET    := $(TARGETDIR)/$(EXECUTABLE)
	LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB)
endif

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

# Add common flags
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)


################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)/$(BINSUBDIR)
COMMONOBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(COMMONFILES)))
SINGLEOBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(SINGLEFILES)))
BENCHOBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(BENCHFILES)))
COCKTAILOBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(COCKTAILFILES)))


################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c.o : $(SRCDIR)/%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)/%.cpp $(C_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

all: makedirectories single bench cocktail

makedirectories:
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(TARGETDIR)
	$(VERBOSE)cp $(ROOTDIR)/kernels/*.cl /tmp/ 
	$(VERBOSE)cp $(ROOTDIR)/include/constant.h /tmp/ 

single: $(COMMONOBJS) $(SINGLEOBJS)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/spmv_all $(COMMONOBJS) $(SINGLEOBJS) $(LIB)

bench: $(COMMONOBJS) $(BENCHOBJS)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_bdia $(OBJDIR)/bench_bdia.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_dia $(OBJDIR)/bench_dia.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_bell $(OBJDIR)/bench_bell.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_sbell $(OBJDIR)/bench_sbell.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_bcsr $(OBJDIR)/bench_bcsr.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_sell $(OBJDIR)/bench_sell.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_ell $(OBJDIR)/bench_ell.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_csr $(OBJDIR)/bench_csr.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_coo $(OBJDIR)/bench_coo.cpp.o $(COMMONOBJS) $(LIB)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/bench_overhead $(OBJDIR)/bench_overhead.cpp.o $(COMMONOBJS) $(LIB)

cocktail: $(COMMONOBJS) $(COCKTAILOBJS)
	$(VERBOSE)$(LINK) -o $(TARGETDIR)/spmv_cocktail $(COMMONOBJS) $(COCKTAILOBJS) $(LIB)


tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(COMMONOBJS) $(SINGLEOBJS) $(BENCHOBJS) $(COCKTAILOBJS)
	$(VERBOSE)rm -f $(TARGETDIR)/*

clobber : clean
	$(VERBOSE)rm -rf $(ROOTOBJDIR)
	$(VERBOSE)find $(TARGETDIR) | egrep "ptx" | xargs rm -f
	$(VERBOSE)find $(TARGETDIR) | egrep "txt" | xargs rm -f
	$(VERBOSE)rm -f $(TARGETDIR)/samples.list

#!/bin/bash
# Script to compile and run a single main.cpp of a benchmark, with gperftools profiling enabled.
# This script is meant to be sourced by another script which sets the appropriate ROOT_FOLDER.
# After using this script it is necessary to compile the library again with OPTFLAGS to have the normal optimized version.

# remember current path
MYPATH="$(pwd)"

# Build the library using the optimization flags
echo ""
echo ""
echo "Building the library with optimization flags..."
echo ""
echo ""
cd "${ROOT_FOLDER}"
./build_profiling_library.sh
cd "${MYPATH}"
echo ""
echo ""
echo "Done."
echo ""
echo ""

source "${ROOT_FOLDER}/config.sh"

FLAG_TO_USE="${GPERFFLAGS}"

rm -f exe
rm -f *.o

#benchmarks/common path
COM_PATH="${ROOT_FOLDER}/benchmarks/common/"

echo ""
echo ""
echo "Building the executable..."
echo ""
echo ""


# Build the main executable
echo "$CC $FLAGS $FLAG_TO_USE -Wall ${FULL_I} -I${COM_PATH} -c *.cpp"
$CC $FLAGS $FLAG_TO_USE -Wall ${FULL_I} -I${COM_PATH} -c *.cpp

case ${OS_NAME} in
    "Darwin")
        echo "$CC $FLAGS $FLAG_TO_USE ${FULL_L} -o exe *.o ${FULL_LIBS}"
        $CC $FLAGS $FLAG_TO_USE ${FULL_L} -o exe *.o ${FULL_LIBS}
        ;;
    "Linux")
        echo "$CC $FLAGS $FLAG_TO_USE ${FULL_L} -Wl,-rpath=${ROOT_FOLDER} -o exe *.o ${FULL_LIBS}"
        $CC $FLAGS $FLAG_TO_USE ${FULL_L} -Wl,-rpath=${ROOT_FOLDER} -o exe *.o ${FULL_LIBS}
        ;;
esac

echo ""
echo ""
echo "Done."
echo ""
echo ""

# Run the executable
echo "Running the executable..."
echo ""
echo ""
LD_PRELOAD=/usr/lib/libtcmalloc.so:/usr/lib/libprofiler.so CPUPROFILE=exe.prof ./exe > benchmark_new.out
echo "Done."
echo ""
echo ""

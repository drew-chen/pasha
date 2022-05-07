all: program

program: cudacode.o
	g++ -std=c++11 -O3 -o created_by_make rand.cpp -fopenmp

cudacode.o:
	nvcc -c -arch=sm_60 cudacode.cu 

clean:
	rm -f *.o program

/**
    Pasha: Parallel Algorithms for Approximating Compact Universal Hitting Sets
    hitting.h
    Header file for main operations for different options of hitting set calculations.
    @author Baris Ekim
    @version 1.0 4/15/19
*/
#ifndef HITTING_H
#define HITTING_H
#include "decycling.h"
#include "graph.h"
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <omp.h>
int graph::Hitting(int L, string hittingFile, int threads) {
/**
Performs hitting set calculations without parallelization
or randomization, counting L-k+1-long paths.
@param L: Sequence length, hittingFile: Output file destination.
@return hittingCount: Size of hitting set.
*/
	vertexExp = pow(ALPHABET_SIZE, k-1);
    int imaxHittingNum = -1;
    ofstream hittingStream;
    int hittingCount = 0;
    l = L-k+1;
    hittingNumArray = new double[edgeNum];
    used = new bool[vertexExp];
	finished = new bool[vertexExp];
	topoSort = new int[vertexExp];
    D = new double*[l + 1];
    double* Dpool = new double[(l+1)* vertexExp];
	for(int i = 0; i < l+1; i++, Dpool += vertexExp) D[i] = Dpool;
	hittingStream.open(hittingFile); 
    F = new double*[l + 1];
    double* Fpool = new double[(l+1)* vertexExp];
	for(int i = 0; i < l+1; i++, Fpool += vertexExp) F[i] = Fpool;
    while (calculatePaths(l, 1)) {
    	int imaxHittingNum = calculateHittingNumber(l);
    	if (imaxHittingNum < 0) break;
        removeEdge(imaxHittingNum);
        string label = getLabel(imaxHittingNum);
        hittingStream << label << "\n";
        hittingCount++;
   	}
   	hittingStream.close();
   	delete [] *D;
	delete [] D;
	delete [] *F;
	delete [] F;
    topologicalSort();
	cout << "Length of longest remaining path: " <<  maxLength() << "\n";
    return hittingCount;
}
int graph::HittingAny(int L, int x, string hittingFile) {
/**
Performs hitting set calculations without parallelization
or randomization, counting paths of all length.
@param L: Sequence length, x: Number of vertices, hittingFile: Output file destination.
@return hittingCount: Size of hitting set.
*/
	vertexExp = pow(ALPHABET_SIZE, k-1);
    ofstream hittingStream;
    int* imaxHittingNum;
    int hittingCount = 0;
    l = L-k+1;
    hittingNumAnyArray = new double[edgeNum];
    used = new bool[vertexExp];
	finished = new bool[vertexExp];
	topoSort = new int[vertexExp];
    hittingStream.open(hittingFile); 
   	topologicalSort();
    D = new double*[1];
    double* Dpool = new double[(1)* vertexExp];
	for(int i = 0; i < 1; i++, Dpool += vertexExp) D[i] = Dpool;
    F = new double*[1];
    double* Fpool = new double[(1)* vertexExp];
	for(int i = 0; i < 1; i++, Fpool += vertexExp) F[i] = Fpool;
    while (maxLength() >= l) { 
    	calculatePathsAny();
    	imaxHittingNum = calculateHittingNumberAny(x);
    	for (int i = 0; i < x; i++) {
	    	removeEdge(imaxHittingNum[i]);
        	string label = getLabel(imaxHittingNum[i]);
        	hittingStream << label << "\n";
	    	hittingCount++;
        }
   	}
   	hittingStream.close();
   	delete [] *D;
	delete [] D;
	delete [] *F;
	delete [] F;
    topologicalSort();
	cout << "Length of longest remaining path: " <<  maxLength() << "\n";
    return hittingCount;
}
int graph::HittingParallel(int L, string hittingFile, int threads) {
/**
Performs hitting set calculations with parallelization
and without randomization, counting L-k+1-long paths.
@param L: Sequence length, hittingFile: Output file destination.
@return hittingCount: Size of hitting set.
*/
	vertexExp = pow(ALPHABET_SIZE, k-1);
    int imaxHittingNum = -1;
    ofstream hittingStream;
    int hittingCount = 0;
    l = L-k+1;
    hittingNumArray = new double[edgeNum];
    stageArray = new int[edgeNum];
    used = new bool[vertexExp];
	finished = new bool[vertexExp];
	topoSort = new int[vertexExp];
    D = new double*[l + 1];
    double* Dpool = new double[(l+1)* vertexExp];
	for(int i = 0; i < l+1; i++, Dpool += vertexExp) D[i] = Dpool;
	hittingStream.open(hittingFile); 
    F = new double*[l + 1];
    double* Fpool = new double[(l+1)* vertexExp];
	for(int i = 0; i < l+1; i++, Fpool += vertexExp) F[i] = Fpool;
    while (calculatePaths(l, threads)) {
    	int imaxHittingNum = calculateHittingNumberParallel(l, false, threads);
    	if (imaxHittingNum < 0) break;
        removeEdge(imaxHittingNum);
        string label = getLabel(imaxHittingNum);
        hittingStream << label << "\n";
        hittingCount++;
   	}
   	hittingStream.close();
    delete [] *D;
	delete [] D;
	delete [] *F;
	delete [] F;
    topologicalSort();
	cout << "Length of longest remaining path: " <<  maxLength() << "\n";
    return hittingCount;
}
int graph::HittingParallelAny(int L, int x, string hittingFile, int threads) {
/**
Performs hitting set calculations with parallelization and
without randomization, counting paths of all length.
@param L: Sequence length, x: Number of vertices, hittingFile: Output file destination.
@return hittingCount: Size of hitting set.
*/
	vertexExp = pow(ALPHABET_SIZE, k-1);
    ofstream hittingStream;
    int hittingCount = 0;
    l = L-k+1;
    hittingNumAnyArray = new double[edgeNum];
    used = new bool[vertexExp];
	finished = new bool[vertexExp];
	topoSort = new int[vertexExp];
   	topologicalSort();
    D = new double*[1];
    double* Dpool = new double[(1)* vertexExp];
	for(int i = 0; i < 1; i++, Dpool += vertexExp) D[i] = Dpool;
    F = new double*[1];
    double* Fpool = new double[(1)* vertexExp];
	for(int i = 0; i < 1; i++, Fpool += vertexExp) F[i] = Fpool;
    hittingStream.open(hittingFile);
    while(maxLength() >= l) { 
    	calculatePathsAny();
    	int* imaxHittingNum;
    	imaxHittingNum = calculateHittingNumberParallelAny(x, threads);
    	for (int i = 0; i < x; i++) {
        	string label = getLabel(imaxHittingNum[i]);
        	removeEdge(imaxHittingNum[i]);
        	hittingStream << label << "\n";
	    	hittingCount++;
        }
   	}
   	hittingStream.close();
   	delete [] *D;
	delete [] D;
	delete [] *F;
	delete [] F;
    topologicalSort();
	cout << "Length of longest remaining path: " <<  maxLength() << "\n";
    return hittingCount;
}
int graph::HittingRandomParallel(int L, string hittingFile, int threads) {
/**
Performs hitting set calculations with parallelization
and with randomization, counting L-k+1-long paths.
@param L: Sequence length, hittingFile: Output file destination.
@return hittingCount: Size of hitting set.
*/
    omp_set_dynamic(0);
    epsilon = 0.0699;
    delta = 0.0599;
    double alpha = 1 - 6*delta - 2*epsilon;
    cout << "Alpha: " << 1/alpha << endl;
    vertexExp = pow(ALPHABET_SIZE, k-1);
    ofstream hittingStream;
    int hittingCount = 0;
    l = L-k+1;
    int i;
    int j;
    hittingNumArray = new double[edgeNum];
    stageArray = new int[edgeNum];
    used = new bool[vertexExp];
	finished = new bool[vertexExp];
	pick = new bool[edgeNum];
	tried = new bool[edgeNum];
	topoSort = new int[vertexExp];
    D = new double*[l + 1];
    double* Dpool = new double[(l+1)* vertexExp];
	for(int i = 0; i < l+1; i++, Dpool += vertexExp) D[i] = Dpool;
	hittingStream.open(hittingFile); 
    F = new double*[l + 1];
    double* Fpool = new double[(l+1)* vertexExp];
	for(int i = 0; i < l+1; i++, Fpool += vertexExp) F[i] = Fpool;
	calculatePaths(l, threads);
	int imaxHittingNum = calculateHittingNumberParallel(l, false, threads);
	h = findLog((1.0+epsilon), hittingNumArray[imaxHittingNum]);
    double prob = delta/l;
    for (int u = h; u > 0; u--){
        total = 0;
    	int hittingCountStage = 0;
    	double pathCountStage = 0;
    	imaxHittingNum = calculateHittingNumberParallel(l, true, threads);
        vector <int> stageVertices = pushBackVector();
		if (imaxHittingNum < 0) break;
        #pragma omp parallel num_threads(threads)
		for (int i : stageVertices) {
        	if ((pick[i] == false) && (hittingNumArray[i] > ((pow(delta, 3)/(1+epsilon)) * total))) {
                stageArray[i] = 0;
				pick[i] = true;
                //removeEdge(i);
                //hittingStream << label << "\n";
    			hittingCountStage++;
    			pathCountStage += hittingNumArray[i];
    		}
            else if ((pick[i] == false)) {
                if (((double) rand() / (RAND_MAX)) <= prob) {
                    //string label = getLabel(v1);
                    stageArray[i] = 0;
                    pick[i] = true;
                    //removeEdge(i);
                    //string label2 = getLabel(v2);
                    //stageArray[j] = 0;
                    //pick[j] = true;
                    //removeEdge(j);
                    //hittingStream << label << "\n" << label2 << "\n";
                    hittingCountStage += 1;
                    pathCountStage += hittingNumArray[i];
                    //pathCountStage += hittingNumArray[j];



                }
            }
    	}
        hittingCount += hittingCountStage;
        if (pathCountStage >= hittingCountStage * pow((1.0 + epsilon), h) * (1 - 6*delta - 2*epsilon)) {
            #pragma omp parallel num_threads(threads)
            for (int i : stageVertices){
                if (pick[i] == true) {
                    removeEdge(i);
                    string label = getLabel(i);
                    hittingStream << label << "\n";
                }
            }
        	h--;
        }
        else hittingCount -= hittingCountStage;
   	}
   	hittingStream.close();
    delete [] *D;
	delete [] D;
	delete [] *F;
	delete [] F;
    topologicalSort();
	cout << "Length of longest remaining path: " <<  maxLength() << "\n";
    return hittingCount;
}

#endif

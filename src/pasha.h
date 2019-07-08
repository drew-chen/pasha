#ifndef PASHA_H
#define PASHA_H
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <iomanip>
#include <cstdint>
#include <omp.h>
using namespace std;
using byte = uint8_t;
class PASHA {
    public:
        byte* finished;
        byte* pick;
        byte* used;
        double delta;
        double epsilon;
        double* hittingNumArray;
        float** D;
        float* Fprev;
        float* Fcurr;
        int ALPHABET_SIZE;
        double edgeCount;
        double edgeNum;
        int k;
        int l;
        int curr;
        int h;
        double count;
        int total;
        int vertexCount; 
        int vertexExp;
        int vertexExp2;
        int vertexExp3;
        int vertexExpMask;
        int vertexExp_1;
        byte* edgeArray;
        byte* stageArray;
        byte* topoSort;
        static map<char, int> alphabetMap;
        string ALPHABET;
        vector<int> stageVertices;
    PASHA (int argK) {
    /**
    Definition of a graph object. Generates a graph of order k, creates an empty
    edge index array, calculates number of edges, builds a character-index map.
    @param argK: Argument passed as k-mer length.
    */
        ALPHABET = "ACGT";
        ALPHABET_SIZE = 4;
        k = argK;
        edgeNum = pow(ALPHABET_SIZE, k);
        edgeArray = new byte[(int)edgeNum];
        generateGraph(k);
        map<char, int> alphabetMap;
        for (int i = 0; i < ALPHABET_SIZE; i++) alphabetMap.insert(pair<char,int>(ALPHABET[i], i));
    }

    void generateGraph(int k) {  
    /**
    Generates a complete de Bruijn graph of order k.
    @param k: Desired k-mer length (order of complete graph).
    */
        for (int i = 0; i < edgeNum; i++) edgeArray[i] = 1;
        edgeCount = edgeNum;
        vertexCount = edgeNum / ALPHABET_SIZE; 
    }

    char getChar(int i) {
    /**
    Gets alphabet character from index.
    @param i: Index of character.
    @return The character in the alphabet.
    */
        return ALPHABET[i];
    }

    string getLabel(int i) {
    /**
    Gets label of the input edge index.
    @param i: Index of edge.
    @return The label of the edge.
    */
        string finalString = "";
        for (int j = 0; j < k; j++) {
            finalString = getChar((i % ALPHABET_SIZE)) + finalString;
            i = i / ALPHABET_SIZE;
        }
        return finalString;
    }
    int maxLength() {
    /**
    Calculates the length of the maximum length path in the graph.
    @return maxDepth: Maximum length.
    */
        vector<int> depth(vertexExp);
        int maxDepth = -1;
        for (int i = 0; i < vertexExp; i++) {
            int maxVertDepth = -1;
            for (int j = 0; j < ALPHABET_SIZE; j++) {
                int edgeIndex = topoSort[i] + j * vertexExp;
                int vertexIndex = edgeIndex / ALPHABET_SIZE;
                if ((depth[vertexIndex] > maxVertDepth) && (edgeArray[edgeIndex] == 1)) maxVertDepth = depth[vertexIndex];
            }
            depth[topoSort[i]] = maxVertDepth + 1;
            if (depth[topoSort[i]] > maxDepth) {maxDepth = depth[topoSort[i]];}
        }
        return maxDepth;
    }

    void removeEdge(int i) {
    /**
    Removes an edge from the graph.
    @param i: Index of edge.
    */
        if (edgeArray[i] == 1) edgeCount--;
        edgeArray[i] = 0;
    }

    void topologicalSort() {
    /**
    Traverses the graph in topological order.
    */
        for (int i = 0; i < vertexExp; i++) {used[i] = false; finished[i] = false;}
        int index = 0;
        for (int i = 0; i < vertexExp; i++) {
            if (used[i] == false) {
                index = depthFirstSearch(index, i);
                if (index == -1) {topoSort = NULL; return;}
            }
        }
       // int rc[vertexExp];
        //for (int i = 0; i < vertexExp; i++) rc[i] = topoSort[vertexExp-i-1];
    }
    int depthFirstSearch(int index, int u) {
    /**
    Depth-first search of a given index of an edge.
    @param index: Depth of recursion, u: Index of edge.
    @return -1: The search cycles, index+1: Current depth.
    */
        used[u] = true;
        bool cycle = false;
        for (int v : getAdjacent(u)) {
            if (used[v] == true && finished[v] == false) cycle = true;
            if (used[v] == false) {
                index = depthFirstSearch(index, v);
                cycle = cycle || (index == -1);
            }
        }
        finished[u] = true;
        topoSort[index] = u;
        if (cycle) return -1;
        else return index + 1;
    }
    vector<int> getAdjacent(int v) {
    /**
    Get adjacent vertices to a given index of a vertex.
    @param v: Index of vertex.
    @return rc: Array of adjacent vertices.
    */
        int count = 0;
        int adjVertex[ALPHABET_SIZE];
        for (int i = 0; i < ALPHABET_SIZE; i++) {
            int index = v + i * vertexExp;
            if (edgeArray[index] == 1) adjVertex[count++] = index / ALPHABET_SIZE;
        }
        vector<int> rc(count);
        for (int i = 0; i < count; i++) {
            rc[i] = adjVertex[i];
        }
        return rc;
    }

    int HittingRandomParallel(int L, const char *hittingPath, int threads) {
    /**
    Performs hitting set calculations with parallelization
    and with randomization, counting L-k+1-long paths.
    @param L: Sequence length, hittingFile: Output file destination.
    @return hittingCount: Size of hitting set.
    */
        omp_set_dynamic(0);
        vertexExp = pow(ALPHABET_SIZE, k-1);
        ofstream hittingStream(hittingPath);
        int hittingCount = 0;
        l = L-k+1;
        delta = 1/(double)l;
        epsilon = (1-8*(delta))/4;
        if (epsilon == 0) epsilon = 0.083333;
        double alpha = 1 - 4*delta -2*epsilon;
        cout << "Alpha: " << 1/alpha << endl;
        cout << "Delta: " << delta << endl;
        cout << "Epsilon: " << epsilon << endl;
        int i;
        int j;
        hittingNumArray = new double[(int)edgeNum];
        stageArray = new byte[(int)edgeNum];
        used = new byte[vertexExp];
        finished = new byte[vertexExp];
        pick = new byte[(int)edgeNum];
        topoSort = new byte[vertexExp];
        D = new float*[l + 1];
        float* Dpool = new float[(l+1)* vertexExp];
        for(int i = 0; i < l+1; i++, Dpool += vertexExp) D[i] = Dpool;
        //hittingStream.open(hittingFile); 
        Fcurr = new float[vertexExp];
        Fprev = new float[vertexExp];
        //double* Fpool = new double[(l+1)* vertexExp];
        //for(int i = 0; i < l+1; i++, Fpool += vertexExp) F[i] = Fpool;
        calculatePaths(l, threads);
        int imaxHittingNum = calculateHittingNumberParallel(l, false, threads);
        cout << "Max hitting number: " << hittingNumArray[imaxHittingNum] << endl;
        h = findLog((1.0+epsilon), hittingNumArray[imaxHittingNum]);
        double prob = delta/(double)l;
        while (calculatePaths(l, threads)) {
            total = 0;
            int hittingCountStage = 0;
            double pathCountStage = 0;
            //calculatePaths(l, threads);
            if (calculateHittingNumberParallel(l, true, threads) < 0) break;
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < (int)edgeNum; i++) {
                #pragma omp critical
                if ((stageArray[i] == 1) && (pick[i] == false) && (hittingNumArray[i] > (pow(delta, 3) * total))) {
                    stageArray[i] = 0;
                    pick[i] = true;
                    hittingCountStage++;
                    pathCountStage += hittingNumArray[i];
                }
            }
            #pragma omp parallel for num_threads(threads) 
            for (int i = 0; i < (int)edgeNum; i++) {
                #pragma omp critical
                if (pick[i] == false && (stageArray[i] == 1)) {
                    if (((double) rand() / (RAND_MAX)) <= prob) {
                            stageArray[i] = 0;
                            pick[i] = true;
                            hittingCountStage += 1;
                            pathCountStage += hittingNumArray[i];
                    }
                }
            }
            hittingCount += hittingCountStage;
            if (pathCountStage >= hittingCountStage * pow((1.0 + epsilon), h) * (1 - 4*delta - 2*epsilon)) {
                for (int i = 0; i < (int)edgeNum; i++) {
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
        topologicalSort();
        cout << "Length of longest remaining path: " <<  maxLength() << "\n";
        return hittingCount;
    }
    //void calculateForEach(int i, int L) {
/**
Calculates hitting number for an edge of specified index with respect to a specified
sequence length, counting paths of length L-k+1.
@param i: Index of edge, L: Sequence length.
*/
        //double hittingNum = 0;
       // for (int j = (1 - edgeArray[i]) * L; j < L; j++) hittingNum = hittingNum + F[j][i % vertexExp] * D[(L-j-1)][i / ALPHABET_SIZE];
        //hittingNumArray[i] = hittingNum;
    //}
    int calculateHittingNumberParallel(int L, bool random, int threads) {
/**
Calculates hitting number of all edges, counting paths of length L-k+1, in parallel.
@param L: Sequence length.
@return imaxHittingNum: Index of vertex with maximum hitting number.
*/  
        omp_set_dynamic(0);
        double maxHittingNum = 0;
        int imaxHittingNum = -1;
        int count = 0;
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < (int)edgeNum; i++) {
            if (random == true) {
                if (((hittingNumArray[i]) >= pow((1.0+epsilon), h-1)) && ((hittingNumArray[i]) <= pow((1.0+epsilon), h))) {
                    stageArray[i] = 1;
                    pick[i] = false;
                    total += hittingNumArray[i] * stageArray[i];
                    count++;
                }
                else {
                    stageArray[i] = 0;
                    pick[i] = false;
                }   
            }
        }
        for (int i = 0; i < (int)edgeNum; i++) {
            if (hittingNumArray[i]*edgeArray[i] > maxHittingNum) {
                maxHittingNum = hittingNumArray[i]; imaxHittingNum = i;
            }
        }
        return imaxHittingNum;
    }


    int calculatePaths(int L, int threads) {
    /**
    Calculates number of L-k+1 long paths for all vertices.
    @param L: Sequence length.
    @return 1: True if path calculation completes.
    */
        omp_set_dynamic(0);
        curr = 1;
        vertexExp2 = vertexExp * 2;
        vertexExp3 = vertexExp * 3;
        vertexExpMask = vertexExp - 1;
        vertexExp_1 = pow(ALPHABET_SIZE, k-2);
        for (int i = 0; i < vertexExp; i++) {D[0][i] = 1; Fprev[i] = 1;}
        for (int j = 1; j <= L; j++) {
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < vertexExp; i++) {
                D[j][i] = edgeArray[i]*D[j-1][(i >> 2)] + edgeArray[i + vertexExp]*D[j-1][((i + vertexExp) >> 2)] + edgeArray[i + vertexExp2]*D[j-1][((i + vertexExp2) >> 2)] + edgeArray[i + vertexExp3]*D[j-1][((i + vertexExp3) >> 2)];
            }
        }
        for (int i = 0; i < (int)edgeNum; i++) hittingNumArray[i] = 0;
        while (curr <= L) {
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < vertexExp; i++) {
                int index = (i * 4);
                Fcurr[i] = edgeArray[index]*Fprev[index & vertexExpMask] + edgeArray[index + 1]*Fprev[(index + 1) & vertexExpMask] + edgeArray[index + 2]*Fprev[(index + 2) & vertexExpMask] + edgeArray[index + 3]*Fprev[(index + 3) & vertexExpMask];
                //cout << Fcurr[i] << endl;
            }
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < (int)edgeNum; i++) {
                hittingNumArray[i] += Fprev[i % vertexExp] * D[(L-curr)][i / ALPHABET_SIZE];
                if (edgeArray[i] == 0) hittingNumArray[i] = 0;
            }
            #pragma omp parallel for num_threads(threads)
            for (int i = 0; i < vertexExp; i++) Fprev[i] = Fcurr[i];
            curr++;
        }
        return 1;
    }
    int findLog(double base, double x) {
    /**
    Finds the logarithm of a given number with respect to a given base.
    @param base: Base of logartihm, x: Input number.
    @return (int)(log(x) / log(base)): Integer logarithm of the number and the given base.
    */
        return (int)(log(x) / log(base));
    }
};
#endif
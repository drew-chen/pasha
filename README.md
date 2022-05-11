<p align="center">
<img align = "center" src ="pasha.png">
</p>
<h1><center>PASHA: A randomized parallel algorithm for efficiently finding near-optimal universal hitting sets</center></h1>

**PASHA** (A randomized parallel algorithm for efficiently finding near-optimal universal hitting sets) is a tool for finding approximations for a small universal hitting set (a small set of k-mers that hits every sequence of length L). A detailed description of the functionality of PASHA, along with results in set size, running time, memory usage and CPU utilization are provided in:

> Ekim, B., Berger, B., Orenstein, Y. A randomized parallel algorithm for efficiently finding near-optimal universal hitting sets. *bioRxiv* (2020).

PASHA is a tool for research and still under development. It is not presented as accurate or free of any errors, and is provided "AS IS", without warranty of any kind.

---
This fork uses CUDA to calculate hitting numbers (number of l-long paths vertexes are a part of).

## Installation and setup

### Installing the package

Clone the repository to your local machine, and compile via `g++` using the commands:

`cd src`<br>
`nvcc -o pasha rand.cpp hitting_num.cu -Xcompiler -fopenmp -lgomp`

### Example commands

To compute the hitting set for a specified k-mer and sequence length, use the command:

`./pasha <kmerlength> <seqlength> <numThreads> <decycPath> <hitPath>`

`usr@usr:src usr$ ./pasha 8 20 24 decyc8.txt hit.txt`<br>

`Decycling set size: 8230`<br>
`Alpha: 2`<br>
`Delta: 0.0769231`<br>
`Epsilon: 0.0961538`<br>
`Max hitting number: 3.71838e+07`<br>
`Length of longest remaining path: 12`<br>
`7.72136 seconds.`<br>
`Hitting set size: 5287`<br>


## License

PASHA is freely available under the [MIT License](https://opensource.org/licenses/MIT).

## References

If you find PASHA useful, please cite the following papers:

- Ekim, B., Berger, B., Orenstein, Y. A randomized parallel algorithm for efficiently finding near-optimal universal hitting sets. *bioRxiv* (2020).

- Orenstein, Y., Pellow, D., Marçais, G., Shamir, R., Kingsford, C. Compact universal k-mer sets. *16th International Workshop on Algorithms in Bioinformatics* (2016).

- Orenstein, Y., Pellow, D., Marçais, G., Shamir, R., Kingsford, C. Designing small universal k-mer hitting sets for improved analysis of high-throughput sequencing. *PLoS Computational Biology* (2017).

## Contributors

Development of PASHA is led by [Barış Ekim](http://people.csail.mit.edu/ekim/), collaboratively in the labs of [Bonnie Berger](http://people.csail.mit.edu/bab/) at the Computer Science and Artificial Intelligence Laboratory (CSAIL) at Massachusetts Institute of Technology (MIT), and [Yaron Orenstein](http://wwwee.ee.bgu.ac.il/~yaronore/) at the Department of Electrical and Computer Engineering at Ben-Gurion University of the Negev (BGU).

## Contact

All of the software executables, manuals, and quick-start scripts, in addition to the calculated universal hitting sets are provided [here](http://pasha.csail.mit.edu/). Should you have any inquiries, please contact [Barış Ekim](http://people.csail.mit.edu/ekim/) at baris [at] mit [dot] edu.



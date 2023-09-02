# hdcomputing-experiments

Hyper-Dimensional Computing experiments (HDComputing, or Vector Symbolic Architectures).

# 1.py

Experimenting with the concept of using a hypervector VSA for data compression. A deterministic codebook is created of an ASCII encoding, and each index position of the given input sequence (or file byte sequence) is bound to the codebook's mapping hypervector. We use level hypervectors for the inputs index because it is a linear representation. This has proven to be beneficial, (see [An Extension to Basis-Hypervectors for Learning
from Circular Data in Hyperdimensional Computing](https://arxiv.org/pdf/2205.07920.pdf) for relevant benchmarks).

Things to try:

* See if permutation can enhance accuracy.
* Optimise allocations & parallelize binding (custom CUDA kernel or CuPy).

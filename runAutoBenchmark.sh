#!/bin/bash
# first run the benchmark with faiss-cpu
source install_faiss_cpu.sh
python3 autoBenchmark.py
# then run the benchmark with faiss-gpu
source install_faiss_gpu.sh
python3 autoBenchmark.py
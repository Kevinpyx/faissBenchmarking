#!/bin/bash
# first uninstall faiss-gpu
echo 'Y' | pip uninstall faiss-gpu
# install faiss-cpu
pip install faiss-cpu

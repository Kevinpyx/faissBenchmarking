### Current Problems
The *_GPU.h5 files in results did not actually use gpu except for FlatL2 due to my misunderstanding. Only FlatL2_10M_96_10k_100_GPU.h5 actually used GPU. The rest of them are using CPU.


### Next steps
Add support to indexes with 3 parameters (e.g. `IVFPQ`)
Automatically measure GPU memory usage when use_gpu = True (done)


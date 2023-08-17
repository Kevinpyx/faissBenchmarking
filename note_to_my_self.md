### Current Problems
The *_GPU.h5 files in results did not actually use gpu except for FlatL2 due to my misunderstanding. Only FlatL2_10M_96_10k_100_GPU.h5 actually used GPU. The rest of them are using CPU. -- SO i deleted them


### Next steps
Change the ipynb file so that they just compare the results with CPU for 10M, 25M, 50M
Add support to indexes with 3 parameters (e.g. `IVFPQ`)
Automatically measure GPU memory usage when use_gpu = True (done)


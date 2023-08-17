# This file wraps 'main.py' into a function and runs it for different indexes and datasizes automatically.
# Please use this file along with runAotoBenchmark.sh
import dataI_O as io
import benchmark as bm

# the function that wraps 'main.py'
def autoBenchmark(method, k, data_size, size_file, dataset, ground_truth, query_set, use_gpu=True, run=1):
    # print whether we are using GPU or not and set file suffix
    gpu_core =  bm.check_gpu()
    if use_gpu and gpu_core > 0:
        print('Using ' + str(gpu_core) + ' GPU cores')
        gpu = "_GPU"
    else:
        print('Using CPU')
        gpu = ""

    

    # Read data
    print('Fetching dataset')
    xb = io.read_fbin(dataset, chunk_size=data_size) # the whole dataset
    num, dim = xb.shape # the number of vectors in the dataset and the dimension of the vectors

    xq = io.read_fbin(query_set) # read the query vectors (for ground truth)

    GT_id, _ = io.read_ground_truth(ground_truth) # GT means ground truth
    # GT = np.dstack((GT_id, GT_dist)) # GT.shape = query_size * k * 2
    print('Data loaded\n...........................')
    print('Dataset size: ' + size_file)

    result_dict = bm.runBenchmark(method, xb, xq, GT_id, k, use_gpu=use_gpu, run=run)

    # Save results
    print('Saving results')
    filename = '/home/ypx/faissTesting/benchmark/results/' + method + '_' + size_file + '_' + str(dim) + '_' + '10k' + '_' + str(k) + gpu + '.h5'
    io.save_results(filename, result_dict['results'])
    print('Results saved as ' + filename + '\n...........................')

################# Main #################

# Define dataset and ground truth to be used here
deep10M = '/home/ypx/faissTesting/dataset/yandex_deep/base.10M.fbin'
deep1B = '/home/ypx/faissTesting/dataset/yandex_deep/base.1B.fbin'
ground_truth_10M_10M = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground_truth_10M_10M' # I made the ground truth according to the dataset base.10M.fbin and query set query.public.10K.fbin
ground_truth_25M_1B = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground_truth_25M_1B'
ground_truth_50M_1B = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground_truth_50M_1B'
query_set = '/home/ypx/faissTesting/dataset/yandex_deep/query.public.10K.fbin'

# Search specifications
# method: currently only 'FlatL2', 'PQ', 'LSH', 'HNSWFlat', 'IVFFlat', 'IVFPQ' are defined  
# k: 1~100
method_lsit = ['FlatL2', 'PQ', 'LSH', 'HNSWFlat', 'IVFFlat', 'IVFPQ']
k = 100
data_size = 10 * 10**6
size_file = '10M'
use_gpu = True
run = 3

# Run autoBenchmark for all methods with same size
'''
for method in method_lsit:
    autoBenchmark(method, k, data_size, size_file, dataset=deep1B, ground_truth=ground_truth_25M_1B, query_set=query_set)
'''

'''
# FlatL2 at different sizes 
autoBenchmark('FlatL2', k, data_size=10*10**6, size_file='10M', dataset=deep10M, ground_truth=ground_truth_10M_10M, query_set=query_set, use_gpu=use_gpu, run=run)

autoBenchmark('FlatL2', k, data_size=25*10**6, size_file='25M', dataset=deep1B, ground_truth=ground_truth_25M_1B, query_set=query_set, use_gpu=use_gpu, run=run)

autoBenchmark('FlatL2', k, data_size=50*10**6, size_file='50M', dataset=deep1B, ground_truth=ground_truth_50M_1B, query_set=query_set, use_gpu=use_gpu, run=run)
'''

'''
# LSH at 10M using gpu
autoBenchmark('LSH', k, data_size=10*10**6, size_file='10M', dataset=deep10M, ground_truth=ground_truth_10M_10M, query_set=query_set, use_gpu=use_gpu, run=run)
'''

run = 1
# IVFFlat and IVGPW at 10M with and without gpu
method = 'IVFFlat'
autoBenchmark(method, k, data_size=10*10**6, size_file='10M', dataset=deep10M, ground_truth=ground_truth_10M_10M, query_set=query_set, use_gpu=True, run=run)
autoBenchmark(method, k, data_size=10*10**6, size_file='10M', dataset=deep10M, ground_truth=ground_truth_10M_10M, query_set=query_set, use_gpu=False, run=run)
method = 'IVFPQ'
# autoBenchmark(method, k, data_size=10*10**6, size_file='10M', dataset=deep10M, ground_truth=ground_truth_10M_10M, query_set=query_set, use_gpu=True, run=run) # this is done
autoBenchmark(method, k, data_size=10*10**6, size_file='10M', dataset=deep10M, ground_truth=ground_truth_10M_10M, query_set=query_set, use_gpu=False, run=run)
# IVFFlat and IVGPW at 25M with and without gpu
method = 'IVFFlat'
autoBenchmark(method, k, data_size=25*10**6, size_file='25M', dataset=deep1B, ground_truth=ground_truth_25M_1B, query_set=query_set, use_gpu=True, run=run)
autoBenchmark(method, k, data_size=25*10**6, size_file='25M', dataset=deep1B, ground_truth=ground_truth_25M_1B, query_set=query_set, use_gpu=False, run=run)
method = 'IVFPQ'
autoBenchmark(method, k, data_size=25*10**6, size_file='25M', dataset=deep1B, ground_truth=ground_truth_25M_1B, query_set=query_set, use_gpu=True, run=run)
autoBenchmark(method, k, data_size=25*10**6, size_file='25M', dataset=deep1B, ground_truth=ground_truth_25M_1B, query_set=query_set, use_gpu=False, run=run)
# IVFFlat and IVGPW at 50M with and without gpu
method = 'IVFFlat'
autoBenchmark(method, k, data_size=50*10**6, size_file='50M', dataset=deep1B, ground_truth=ground_truth_50M_1B, query_set=query_set, use_gpu=True, run=run)
autoBenchmark(method, k, data_size=50*10**6, size_file='50M', dataset=deep1B, ground_truth=ground_truth_50M_1B, query_set=query_set, use_gpu=False, run=run)
method = 'IVFPQ'
autoBenchmark(method, k, data_size=50*10**6, size_file='50M', dataset=deep1B, ground_truth=ground_truth_50M_1B, query_set=query_set, use_gpu=True, run=run)
autoBenchmark(method, k, data_size=50*10**6, size_file='50M', dataset=deep1B, ground_truth=ground_truth_50M_1B, query_set=query_set, use_gpu=False, run=run)

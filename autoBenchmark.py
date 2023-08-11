# This file wraps 'main.py' into a function and runs it for different indexes and datasizes automatically.
import dataI_O as io
import benchmark as bm

# the function that wraps 'main.py'
def autoBenchmark(method, k, data_size, size_file, dataset, ground_truth, query_set, run=1):
    # print whether we are using GPU or not and set file suffix
    gpu_core =  bm.check_gpu()
    if gpu_core:
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
    print('Database size: ' + str(num))

    result_dict = bm.runBenchmark(method, xb, xq, GT_id, k, run=run)

    # Save results
    print('Saving results')
    filename = '/home/ypx/faissTesting/benchmark/results/' + method + '_' + size_file + '_' + str(dim) + '_' + '10k' + '_' + str(k) + gpu + '.h5'
    io.save_results(filename, result_dict['results'])
    print('Results saved')

################# Main #################

# Define dataset and ground truth to be used here
deep10M = '/home/ypx/faissTesting/dataset/yandex_deep/base.10M.fbin'
deep1B = '/home/ypx/faissTesting/dataset/yandex_deep/base.1B.fbin'
ground_truth_10M_10M = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground_truth_10M_10M' # I made the ground truth according to the dataset base.10M.fbin and query set query.public.10K.fbin
ground_truth_25M_1B = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground_truth_25M_1B'
ground_truth_50M_1B = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground_truth_50M_1B'
query_set = '/home/ypx/faissTesting/dataset/yandex_deep/query.public.10K.fbin'

# Search specifications
# method: currently only 'FlatL2', 'PQ', 'LSH', 'HNSWFlat' are defined  
# k: 1~100
method_lsit = ['FlatL2', 'PQ', 'LSH', 'HNSWFlat']
k = 100
data_size = 25 * 10**6
size_file = '25M'

# Run autoBenchmark
for method in method_lsit:
    autoBenchmark(method, k, data_size, size_file, dataset=deep1B, ground_truth=ground_truth_25M_1B, query_set=query_set)
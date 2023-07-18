import faiss
import dataI_O as io
import struct
import numpy as np
import benchmark as bm

# Define dataset and ground truth to be used here
dataset = '/home/ypx/faissTesting/dataset/yandex_deep/base.10M.fbin'
ground_truth = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground_truth' # I made the ground truth according to the dataset base.10M.fbin and query set query.public.10K.fbin
query_set = '/home/ypx/faissTesting/dataset/yandex_deep/query.public.10K.fbin'

# Search specifications
# method: currently only 'FlatL2', 'PQ', 'LSH', 'HNSWFlat' are defined
# k: 1~100
method = 'FlatL2'
k = 100

# Read data
print('Fetching dataset')
xb = io.read_fbin(dataset) # the whole dataset
num, dim = xb.shape # the number of vectors in the dataset and the dimension of the vectors

xq = io.read_fbin(query_set) # read the query vectors (for ground truth)
query_size = xq.shape[0] # the number of query vectors

#GT_id, GT_dist = io.read_ground_truth(ground_truth) # GT means ground truth
#GT = np.dstack((GT_id, GT_dist)) # GT.shape = query_size * k * 2
GT_id = io.read_ibin(ground_truth)
K = GT_id.shape[1] # the number of nearest neighbors
print('Data loaded\n...........................')

result = bm.runBenchmark(method, xb, xq, GT_id, k)

print(result)
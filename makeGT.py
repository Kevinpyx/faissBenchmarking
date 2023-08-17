import faiss
import numpy as np
import dataI_O as io

def write_ground_truth(xb, xq, k, filename):
    # xb: the dataset
    # xq: the query vectors
    # k: the number of nearest neighbors to be searched
    # the function writes the ground truth to a binary file in the format specified in the benchmark

    index = faiss.GpuIndexFlatL2(xb.shape[1])
    print('building index')
    index.add(xb)
    print('searching...')
    D, I = index.search(xq, k)

    ids = np.array(I, dtype=np.uint32)
    dists = np.array(D, dtype=np.float32)

    # write to binary file
    print('writing to binary file')
    with open(filename, 'wb') as f:
        # write the number of queries and the number of nearest neighbors
        np.array([xq.shape[0], k], dtype=np.uint32).tofile(f)
        # write the ids
        ids.tofile(f)
        # write the distances
        dists.tofile(f)

### Main
data_size = 25 * 10**6 # 25M
size_in_filename = '25M'

filename = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground_truth_' + str(size_in_filename) + '_1B'# meaning 100M vectors our of the 1B vector file
dataset = '/home/ypx/faissTesting/dataset/yandex_deep/base.1B.fbin'
query_set = '/home/ypx/faissTesting/dataset/yandex_deep/query.public.10K.fbin'
print('reading data')
xb = io.read_fbin(dataset, chunk_size=data_size)
print('rading query set')
xq = io.read_fbin(query_set)
k = 100
print('data read')
print('xb.shape = ' + str(xb.shape))
print('xq.shape = ' + str(xq.shape))

write_ground_truth(xb, xq, k, filename)
import faiss
import numpy as np
import dataI_O as io

def write_ground_truth(xb, xq, k, filename):
    # xb: the dataset
    # xq: the query vectors
    # k: the number of nearest neighbors to be searched
    # the function writes the ground truth to a binary file in the format specified in the benchmark

    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    D, I = index.search(xq, k)

    ids = np.array(I, dtype=np.uint32)
    dists = np.array(D, dtype=np.float32)

    # write to binary file
    with open(filename, 'wb') as f:
        # write the number of queries and the number of nearest neighbors
        np.array([xq.shape[0], k], dtype=np.uint32).tofile(f)
        # write the ids
        ids.tofile(f)
        # write the distances
        dists.tofile(f)



filename = '/home/ypx/faissTesting/dataset/yandex_deep/my_ground' 
dataset = '/home/ypx/faissTesting/dataset/yandex_deep/base.10M.fbin'
query_set = '/home/ypx/faissTesting/dataset/yandex_deep/query.public.10K.fbin'
xb = io.read_fbin(dataset)
xq = io.read_fbin(query_set)
k = 100

write_ground_truth(xb, xq, k, filename)
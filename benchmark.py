import time
import faiss
import numpy as np
import itertools # for cartesian product
import psutil # for memory usage

# macro to indicate the number of indexing methods tested
INDNUM = 4

# a nested dictionary for indexing methods and their parameter values that will be used for testing for 
# different indexing methods (every method needs the dimension of the vectors)
# keys: index names
# values: dictionary that stores Indexing method, number of parameter, values of parameters
INDPARAM = {
    'FlatL2' : {
        'Index' : 'FlatL2',
        'params' : [], # empty list because no parameters 
        # no parameters other than dimension
        'results' : []
    },
    'PQ' : { # product quantization
        'Index' : 'PQ',
        'params' : ['m', 'nbits'],
        'm' : [1, 2, 3, 8, 12, 16, 24, 32, 48, 96], # number of subspaces
        'nbits' : [6, 8, 10], # 2^n is number of centroid for every subspace
        'results' : [] # the results will be stored in this list
    }, 
    'LSH' : {
        'Index' : 'LSH',
        'params' : ['nbits'],
        'nbits' : [2, 6, 8, 16, 24, 32, 64], # the number of hyperplanes to used
        'results' : []
    },
    'HNSWFlat' : {
        'Index' : 'HNSWFlat',
        'params' : ['M'],
        'M' : [32, 64], # the number of nearest neighbor connections every vertex in the constructed graph has
        'results' : []
    }
}

INDLIST = list(INDPARAM.keys())

# Train and add data to indexes
#   precon: 
#       - index is a faiss index
#       - xb is a vector database (2-D array with rows of vectors)
#   postcon:
#       - index will be trained
def train_index(index, xb):
    print("Training index...")
    time0 = time.time()
    index.train(xb)
    time1 = time.time()
    print("Adding data...")
    index.add(xb)
    time2 = time.time()
    return time1-time0, time2-time1 # return training time and adding time

# Perform search
#   precon: 
#       - index is a trained faiss index
#       - xq are query vectors (2-D array rows of vectors)
#       - k is a nonnegative integer to indicate how many nearest neighbors to find
#   postcon: 
#       - two 2-D arays will be returned
#       - the first variable will be the distances from the query vectors to k nearest vectors found
#       - the second variable will be the ids of k nearest vectors found
#       - the shape of both arrays is number of queries * k
def search_index(index, xq, k):
    return index.search(xq, k)


# return all possible parameter combinations in a list of tuples
#   precon:
#       - method is a valid faiss indexing method
def get_parameters(method):
    index_param = INDPARAM[method] # get dictionary for the specific index
    print('Index used:', method) # show index method
    parameters = index_param['params'] # list of strings
    param_list = parameters.copy()
    param_list.insert(0, 'dimension') # create a separate list of parameters to display (with dimension as one parameter)
    numParam = len(parameters)
    print('parameters:', param_list) # show parameters being tested

    # build parameter combinations
    if numParam > 1:
        # get a list of lists of parameter values
        param_valuess = [index_param[param] for param in parameters] # param_lists is in the form of [[values for parameter 1], [values for parameter 2], ...]
        # Generate the cartesian product combinations to get all parameter combinations
        combinations = itertools.product(*param_valuess)
        # Convert the combinations to a list
        param_combinations = list(combinations)
    elif numParam == 1:
        param_combinations = [(x,) for x in index_param[parameters[0]]] # combination is just a list of singleton tuples as values of the single parameter
    else:
        param_combinations = None # no parameter

    return param_combinations

# a helper function for get_indeces to determine which index to use according to method
def create_index(method):
    if method == 'FlatL2':
        def IndexFlatL2(d):
            return faiss.IndexFlatL2(d)
        return IndexFlatL2
    elif method == 'PQ': 
        def IndexPQ(d, m, nbits):
            return faiss.IndexPQ(d, m, nbits)
        return IndexPQ
    elif method == 'LSH':
        def IndexLSH(d, nbits):
            return faiss.IndexLSH(d, nbits)
        return IndexLSH
    elif method == 'HNSWFlat':
        def IndexHNSWFlat(d, M):
            return faiss.IndexHNSWFlat(d, M)
        return IndexHNSWFlat
    
# return the algorithms to benchmark with different values received
#   precon: 
#       - value should be an integer between 1 and 3 inclusive
#   postcon: 
#       - the frist variable will hold the faiss index object corresponsding to the value
#       - the second variable is a nonnegative integer that indicates how many main parameters the method needs
#       - the third variable is the name of the index/algorithm in a string
def get_indices(method, d, param_combinations):
    faissIndex = create_index(method) # getting the corresponding index constructor
    indList = []
    numParam = 1 # the smallest number of parameters is 1 (the dimension)
    if param_combinations == None:
        pass
    else:
        numParam += len(param_combinations[0])

    # building indices
    if numParam == 1:
        indList.append(faissIndex(d))
    elif numParam == 2:
        for x in param_combinations:
            indList.append(faissIndex(d, x[0]))
    elif numParam == 3:
        for x in param_combinations:
            indList.append(faissIndex(d, x[0], x[1]))
    else:
        print('Parameter number error')
        return None
    
    return indList # list of indices


# precon: 
#       - GT_id and pred_id are 2-D arrays with the same shape
#       - the first dimension is the number of queries
#       - the second dimension is the number of nearest neighbors
# postcon:
#       - the hit rate will be returned (from 0 to 1)

def get_accuracy(GT_id, pred_id): #calculate the hit rate
    num_queries, knn = GT_id.shape
    hit_count = 0

    for i in range(num_queries):
        GT_set = set(GT_id[i, :])
        pred_set = set(pred_id[i, :])
        intersection = GT_set.intersection(pred_set)
        hit_count += len(intersection)

    return hit_count/np.prod(GT_id.shape) # the hit rate = hit count/number of elements in GT

def measure_memory_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)

# runBenchmark: 
#   precon: 
#       - method must be one of the faiss index defined in INDPARAM
#       - k must be an integer between 1 and k_max in the ground truth
#       - GT.shape = query_size * k_max * 2 (first layer: ids, second layer: distances)

def runBenchmark(method, xb, xq, GT_id, k=None, run=1): 
    # check whether method is valid
    if method in INDLIST:
        pass
    else:
        print('Invalid index. Available indexing methods:', INDLIST)
        return None

    query_size, dim = xq.shape # the number of query vectors and the dimension of the vectors

    # determining k value (because we don't have to compare all the k_max nearest neighbor in the ground truth if we don't want to)
    k_max = GT_id.shape[1] # the number of nearest neighbors in ground truth
    if (k is None or k >= k_max): 
        k = k_max # if k is not specified or more than max k in ground truth, we compare all the query_size *k vectors we have
        #GT_id = GT_id[:,:,0]
        #GT_dist = GT[:,:,1]
    elif (k > 0): # k has a valid value
        # we unpack and trim the ground truth according to k
        GT_id = GT_id[:, :k]
        #GT_dist = GT[:, :k, 1]
    else: # k is smaller than 0
        print("k must be between 1 and", k_max)
        return None

    # get all the different parameter combinations 
    param_combinations = get_parameters(method)

    # get faiss indices instantiated with different parameter combinations
    indList = get_indices(method, dim, param_combinations)
    if indList == None:
        print('Unable to get faiss indices')
        return None
    
    result_dict = INDPARAM[method] # we will append the results to this dictionary and return so that users can see the parameters used for each result
    # prepare these numbers outside of the loop to avoid repeated calculation
    parameters = result_dict['params'] # list of parameters
    numParam = len(parameters) # number of parameters

    for turn in range(run): # run the benchmark for the specified number of times
        print('Run', turn+1, 'out of', run)

        # prepare for the loop
        total_rounds = 1 if param_combinations is None else len(param_combinations) # number of rounds
        round_number = 1 # the round number

        # prepare lists for result storage
        result_num = 6 # modify when you add or remove
        training_time = []
        adding_time = []
        total_search_time = []
        time_per_vec = []
        memory = []
        hit_rates = []
    
        # for every index, we train and perform timed search
        for ind in indList:
            if param_combinations is None:
                print('Round 1 / 1: Index', method, 'with parameter', dim) 
            else:
                print('Round', round_number, '/', total_rounds, ': Index', method, 'with parameters', dim, *param_combinations[round_number-1])

            # train the index and get time spent
            t_t, t_a = train_index(ind, xb)
            training_time.append(t_t)
            adding_time.append(t_a)

            # start the timer
            start_time = time.time()

            # perform search
            print('Searching for', k, 'nearest neighbors...')
            memory.append(measure_memory_usage())
            pred_dist, pred_id = search_index(ind, xq, k)

            # end the timer
            end_time = time.time()

            # calculate and append the elapsed time
            elapsed = end_time - start_time
            total_search_time.append(elapsed)
            per_vec = elapsed/(query_size * k)
            time_per_vec.append(per_vec)

            # calculate hit rate
            hit_rates.append(get_accuracy(GT_id, pred_id))
            
            # increment the round number
            round_number += 1

            ind.reset() # reset the index

        # combine, format, and return the time and hitrate result
        results = np.dstack((training_time, adding_time, total_search_time, time_per_vec, memory, hit_rates))

        # 
        print('Results for', method,'in run', turn+1, ':', results)

        if numParam < 2:
            result_dict['results'].append(results)
        else: 
            x = [len(result_dict[parameters[n]]) for n in range(numParam)] # list of numbers of parameter values (e.g. for PQ, m = [8, 16], nbits = [8, 12, 16], x = [2, 3])
            x.append(result_num) 
            result_dict['results'].append(results.reshape(tuple(x))) # reshape the results to the shape of the parameter values and add to the dictionary

    return result_dict



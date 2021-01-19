#cython: boundscheck=False, wraparound=False
cimport cython
import numpy as np
cimport numpy as np
import random
from scipy.sparse import csr_matrix
from cython.parallel import prange
from sklearn.svm import LinearSVC

from libc.math cimport log, abs, exp, pow, sqrt
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdlib cimport malloc, free
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp cimport bool
from libcpp.algorithm cimport sort as stdsort
from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef pair[vector[int],vector[int]] LR_SET
ctypedef pair[int,int] I_PAIR
ctypedef pair[int,float] DP
ctypedef vector[I_PAIR] COUNTER
ctypedef vector[vector[int]] YSET
ctypedef vector[pair[int,float]] SR
ctypedef vector[SR] CSR
ctypedef vector[vector[float]] Mat


cdef float cal_cosine(float *a, const SR &b):
    cdef int n = b.size(), i = 0
    cdef float prod = 0
    for i in range(n):
        prod += a[b[i].first] * b[i].second
    return prod

cdef void add_sparse(float *a, const SR &b):
    cdef int n = b.size(), i = 0
    for i in range(n):
        a[b[i].first] += b[i].second

cdef float get_norm(float *a, int n):
    cdef float norm = 0
    cdef int i = 0
    for i in range(n):
        norm += a[i] * a[i]
    return sqrt(norm)

cdef void div_by_scalar(float *a, int n, float s):
    cdef int i = 0
    for i in range(n):
        a[i] /= s

cdef void normalize(float *a, int n):
    cdef float s = get_norm(a, n)
    if s > 0:
        div_by_scalar(a, n, s)

cdef int get_rand_num(int siz):
    return random.randint(0, siz - 1)

cdef eliminate_zeros(vector[int] &indices, vector[float] &data):
    cdef int sz = data.size(), i = 0, j = 0
    for i in range(sz):
        if data[i] != 0:
            indices[j] = indices[i]
            data[j] = data[i]
            j += 1
    indices.resize(j)
    data.resize(j)

cpdef (vector[int], vector[int], vector[float]) random_swap(const vector[int]& indices, const vector[int]& indptr, vector[float]& data, int n):
    #print ('random swap begins..')
    random.seed(1234)
    cdef int i = 0, j = 0, m = 0
    cdef int N = indptr.size() - 1
    cdef int idx_1, idx_2, counter = 0
    for i in range(N):
        #print (f'# swap {i}')

        m = indptr[i+1] - indptr[i]
        for j in range(n):
            if m <= 1:
                break
            #counter = 0
            idx_1 = get_rand_num(m) + indptr[i]
            idx_2 = idx_1
            while idx_1 == idx_2:
                idx_2 = get_rand_num(m) + indptr[i]
                counter += 1
                if counter > 3:
                    break
            data[idx_1], data[idx_2] = data[idx_2], data[idx_1]

    #print ('random swap ends..')
    return indices, indptr, data

cpdef (vector[int], vector[int], vector[float]) random_deletion(const vector[int]& indices, const vector[int]& indptr, vector[float]& data, float p):
    cdef int i = 0, j = 0, m = 0
    cdef int N = indptr.size() - 1
    cdef float r

    for i in range(N):
        m = indptr[i + 1] - indptr[i]
        if m <= 1:
            continue
        for j in range(indptr[i], indptr[i+1]):
            r = random.uniform(0, 1)
            if r < p:
                data[j] = 0

    #eliminate_zeros(indices, data)
    return indices, indptr, data

cpdef vec_to_np(const vector[int] &a):
    cdef int n = a.size(), i = 0
    cdef np.ndarray[np.int32_t] b = np.zeros(n, dtype=np.int32)
    for i in range(n):
        b[i] = a[i]
    return b

cpdef vec_to_npf(const vector[float] &a):
    cdef int n = a.size(), i = 0
    cdef np.ndarray[np.float32_t] b = np.zeros(n, dtype=np.float32)
    for i in range(n):
        b[i] = a[i]
    return b

cdef vector[int] argsort_vec(const vector[float] &a):
    #cdef np.ndarray[np.float32_t] b = vec_to_npf(a)
    cdef vector[int] res = np.argsort(a)
    return res

cdef list vec_to_list2d(const vector[vector[int]] &a):
    cdef unsigned int n = a.size()
    cdef list L = [None] * n
    cdef unsigned int i, j, m
    for i in range(n):
        m = a[i].size()
        L[i] = [0] * m
        for j in range(m):
            L[i][j] = a[i][j]
    return L

cdef list vec_to_list2df(const vector[vector[float]] &a):
    cdef unsigned int n = a.size()
    cdef list L = [None] * n
    cdef unsigned int i, j, m
    for i in range(n):
        m = a[i].size()
        L[i] = [0] * m
        for j in range(m):
            L[i][j] = a[i][j]
    return L

cpdef vector[vector[int]] group_by_value(const vector[int] &labels):
    cdef vector[vector[int]] res
    #cdef unordered_map[int, int] vis
    cdef int i, j = 0, n = labels.size()
    cdef vector[int] vis
    cdef int n_clusters = 110
    vis.resize(n_clusters)
    for i in range(n_clusters):
        vis[i] = -1
    for i in range(n):
        #if vis.find(labels[i]) == vis.end():
        if vis[labels[i]] == -1:
            vis[labels[i]] = j
            j += 1
    res.resize(j)
    for i in range(n):
        res[vis[labels[i]]].push_back(i)
    #return res
    return vec_to_list2d(res)

cpdef list one_vs_all(object X, object Y, int rs, float th=0):
    cdef int n_labels = Y.shape[1]
    cdef int n_features = X.shape[1]
    cdef int n_samples = X.shape[0]
    cdef int i, j
    cdef list W = [None] * n_labels
    cdef vector[short] y
    cdef vector[int] pos
    y.resize(n_samples)
    for i in range(n_samples):
        y[i] = -1

    for i in range(n_labels):
        pos = Y[:, i].indices
        sz = pos.size()
        for j in range(sz):
            y[pos[j]] = 1

        if sz == 0:
            w = [0] * (n_features + 1)
        elif sz == n_samples:
            w = [0] * (n_features)
            w.append(1)
        else:
            svm = LinearSVC(max_iter=20, random_state=rs).fit(X, y)
            w = np.concatenate((svm.coef_[0], svm.intercept_))
            # for large datasets only
            #w[w < th] = 0
            #######################
        W[i] = csr_matrix(w)

        for j in range(sz):
            y[pos[j]] = -1
    return W


cpdef updateIP(const int sz, const int N, const vector[int]& indices, const vector[int]& indptr, const vector[float]& data, int topk=5):
    cdef vector[vector[int]] I
    cdef vector[vector[float]] P
    cdef int i = 0, j = 0, m = 0, k,
    cdef vector[float] cur_data
    cdef vector[int] idx

    cdef int ptr = 0
    I.resize(sz)
    P.resize(sz)
    for i in range(N):
        if indptr[i+1] == indptr[i]:
            continue

        m = indptr[i+1] - indptr[i]
        if m > topk:
            cur_data.resize(m)
            for j in range(m):
                cur_data[j] = data[j+indptr[i]]
            #idx = argsort_vec(cur_data)
            idx = np.argsort(cur_data)
        else:
            idx = np.arange(m)
            topk = m

        for j in range(topk):
            k = idx[m - 1 - j]
            I[indices[indptr[i]+k]].push_back(i)
            P[indices[indptr[i]+k]].push_back(data[indptr[i]+k])
    #return I, P
    return vec_to_list2d(I), vec_to_list2df(P)


cpdef vector[int] updateR(const vector[int]& c, const vector[int]& indices, const vector[int]& indptr):
    cdef int n = c.size()
    cdef int i = 0, j = 0
    cdef vector[int] res
    cdef unordered_set[int] vis
    #res.reserve(n)
    for i in range(n):
        for j in range(indptr[c[i]], indptr[c[i]+1]):
            if vis.find(indices[j]) == vis.end():
                res.push_back(indices[j])
                vis.insert(indices[j])
    #return res
    return vec_to_np(res)


cpdef (vector[int], vector[int], vector[float]) updateRCD(const vector[int]& idx, const vector[int]& labels, const vector[int]& indices, const vector[int] indptr, const vector[float]& data, const vector[float]& p, int doexp=0):
    #print ('updateRCD starting...')
    cdef int n = idx.size(), i = 0, j = 0, k
    cdef unsigned int m = data.size()
    cdef vector[int] r, c, tmp_idx
    cdef vector[float] d, tmp
    cdef int topk = 10
    r.reserve(m)
    c.reserve(m)
    d.reserve(m)

    for i in range(n):
        tmp.clear()
        for k in range(indptr[i], indptr[i+1]):
            tmp.push_back(data[k] + p[i])
        tmp_idx = np.argsort(tmp)
        topk = min(topk, tmp.size())
        if doexp:
            topk = tmp.size()
        for k in range(topk):
            j = tmp_idx[tmp.size() - 1 - k] + indptr[i]
        #while j < indptr[i + 1]:
            r.push_back(idx[i])
            c.push_back(labels[indices[j]])
            d.push_back(data[j] + p[i])
            if doexp:
                d[j] = exp(d[j])
            #j += 1
    #print ('updateRCD ends...')
    #return r, c, d
    return vec_to_np(r), vec_to_np(c), vec_to_npf(d)

cpdef vector[int] kmeans(A, float eps, int K, int rs):
    random.seed(rs)
    #print (A.shape)
    cdef int num_sample = A.shape[0]
    cdef int num_feature = A.shape[1]
    cdef int i, j, m, p
    cdef CSR X
    csr_to_vec(A, X)
    #cdef vector[vector[float]] centers = np.zeros((K, num_feature))
    cdef float **centers = <float **>malloc(K * sizeof(float *))
    for i in range(K):
        centers[i] = <float *>malloc(num_feature * sizeof(float))
    cdef float **cosines = <float **>malloc(K * sizeof(float *))
    for i in range(K):
        cosines[i] = <float *>malloc(num_sample * sizeof(float))
    for i in range(K):
        for j in range(num_feature):
            centers[i][j] = 0

    #cdef vector[vector[float]] cosines = np.zeros((K, num_sample))
    cdef vector[int] partition
    partition.resize(num_sample)

    cdef vector[int] perm = np.arange(num_sample)
    np.random.shuffle(perm)

    ###### Initialization centers begins ########
    for i in range(num_sample):
        p = get_rand_num(K)
        add_sparse(centers[p], X[i])
    for i in range(K):
        normalize(centers[i], num_feature)
        #p = get_rand_num(num_sample)
        #add_sparse(centers[i], X[perm[i]])
    ###### Initialization centers ends ########

    cdef float old_cos = -10000, new_cos = -1
    cdef float best_sim = 0
    cdef int best_center = 0
    cdef int it = 0
    while new_cos - old_cos >= eps:
        it += 1
        for i in range(K):
            for j in range(num_sample):
                cosines[i][j] = cal_cosine(centers[i], X[j])

        old_cos = new_cos
        new_cos = 0
        for i in range(num_sample):
            best_sim = 0
            best_center = partition[i]
            for j in range(K):
                if cosines[j][i] > best_sim:
                    best_sim = cosines[j][i]
                    best_center = j

            partition[i] = best_center
            new_cos += best_sim

        new_cos /= num_sample

        #reset centers
        for i in range(K):
            for j in range(num_feature):
                centers[i][j] = 0

        for i in range(num_sample):
            add_sparse(centers[partition[i]], X[i])
        for i in range(K):
            normalize(centers[i], num_feature)

    print (f'kmeans finished in {it} iterations')
    cdef vector[int] count = np.zeros(K)
    for i in range(num_sample):
        count[partition[i]] += 1

    #cdef bool changed = False
    for i in range(num_sample):
        if count[partition[i]] == 1:
            best_center = 0
            best_sim = 0
            for j in range(K):
                if cosines[j][i] > best_sim and j != partition[i]:
                    best_sim = cosines[j][i]
                    best_center = j
            #count[partition[i]] -= 1
            partition[i] = best_center
            count[best_center] += 1

    for i in range(K):
        free(centers[i])
        free(cosines[i])
    free(centers)
    free(cosines)

    squeeze_vec(partition)
    #return partition
    cdef np.ndarray[np.int32_t] labels = np.zeros(num_sample, dtype=np.int32)
    for i in range(num_sample):
        labels[i] = partition[i]
    return labels

cpdef void squeeze_vec(vector[int] &a):
    cdef int n = a.size()
    cdef int i = 0, j = 0
    cdef unordered_map[int, int] reid
    for i in range(n):
        if reid.find(a[i]) == reid.end():
            reid[a[i]] = j
            j += 1
        a[i] = reid[a[i]]


cpdef object vec_to_csr(CSR& csr, const int N, const int M):
    '''
    convert CSR to csr_matrix
    '''
    cdef int n = csr.size()
    cdef int i, j, m
    cdef int nnz = 0
    for i in range(n):
        nnz += csr[i].size()

    cdef np.ndarray[np.int32_t] row = np.zeros(nnz, dtype=np.int32)
    cdef np.ndarray[np.int32_t] col = np.zeros(nnz, dtype=np.int32)
    cdef np.ndarray[np.float32_t] data = np.zeros(nnz, dtype=np.float32)

    cdef int ptr = 0
    for i in range(n):
        m = csr[i].size()
        for j in range(m):
            row[ptr], col[ptr], data[ptr] = i, csr[i][j].first, csr[i][j].second

    res = csr_matrix((data, (row, col)), shape=(N, M))
    return res

cpdef void csr_to_vec(object A, CSR &csr):
    '''
    convert csr_matrix to CSR
    '''
    cdef int i, j = 0
    cdef vector[int] ind = A.indices, indptr = A.indptr
    cdef vector[float] data = A.data
    cdef int n = indptr.size() - 1
    csr.resize(n)

    for i in range(n):
        csr[i].resize(indptr[i+1]-indptr[i])
        while j < indptr[i + 1]:
            assert ind[j] < A.shape[1]
            csr[i][j - indptr[i]].first = ind[j]
            csr[i][j - indptr[i]].second = data[j]
            j += 1

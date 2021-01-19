
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix, find, vstack, csc_matrix, hstack
from util import timeit
from sklearn.svm import LinearSVC
from scipy.sparse.linalg import norm
import kmeans as ckmeans
import pickle

class Node:
    def __init__(self):
        self.isLeaf = False

    def classifierNorm(self, w, p=1):
        return w / norm(w)
        #return w / (np.sqrt(np.sum(w ** 2)) ** p)

    def getNorm(self):
        return (self.labels, norm(self.w, axis=1))

    @timeit
    def buildNode(self, X, Y, labels, isLeaf=False, rs=0, th=0):
        self.isLeaf = isLeaf
        self.labels = labels
        self.w = ckmeans.one_vs_all(X, Y, rs, th)
        self.w = vstack(self.w)

    def predict(self, X):
        pred = X.dot(self.w.transpose())
        pred.data = -np.maximum(0, 1 - pred.data) ** 2
        return pred

class BonsaiForest:
    def __init__(self, num_tree=3, nc=50, md=2, n_jobs=3, rand_seed=12345, nnC=0, weights=None):
        self.num_tree = num_tree
        self.nc = nc
        self.max_depth = md
        self.n_jobs = n_jobs
        self.rand_seed = rand_seed
        self.nnC = nnC
        self.weights = weights

    @timeit
    def labelRepresentation(self, X, Y):
        # input representation
        self.Ri = Y.transpose().dot(X)
        # row normalization
        row_sums = norm(self.Ri, axis=1)
        row_indices, _ = self.Ri.nonzero()
        self.Ri.data /= row_sums[row_indices]

    @timeit
    def fit(self, X, Y):
        self.num_sample, self.num_feature = X.shape
        _, self.num_label = Y.shape

        self.labelRepresentation(X, Y)

        self.clf = [Bonsai(rand_seed=self.rand_seed*i+2020, nc=self.nc, md=self.max_depth)\
                    for i in range(self.num_tree)]

        Parallel(n_jobs=self.n_jobs, backend="threading")\
            (delayed(self.clf[i].train)(X, Y, self.Ri) for i in range(self.num_tree))

        del self.Ri

    def getNorm(self):
        weight_norm = Parallel(n_jobs=self.n_jobs, backend="threading")\
            (delayed(self.clf[i].traverse)() for i in range(self.num_tree))
        return weight_norm

    def save(self, filename):
        file = open(filename, 'wb')
        pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @timeit
    def predict(self, X, topk=5):
        X = csr_matrix(hstack((X, csr_matrix(np.ones((X.shape[0], 1))))))

        n_samples = X.shape[0]
        offset = 0
        batch_size = 5000

        prob = [[] for _ in range(self.num_tree)]
        while offset < n_samples:
            rb = min(offset+batch_size, n_samples)
            res = Parallel(n_jobs=self.n_jobs, backend="threading")\
                (delayed(self.clf[i].predict)(X[offset:rb], topk) for i in range(self.num_tree))
            for i in range(self.num_tree):
                prob[i].append(res[i][1])

            offset += batch_size
        for i in range(self.num_tree):
            prob[i] = vstack(prob[i])
        prob = csr_matrix(sum(prob)/self.num_tree)
        return None, prob

class Bonsai:
    def __init__(self, rand_seed, nc, md):
        self.root = None
        self.rand_seed = rand_seed
        self.max_depth = md
        self.num_cluster = nc

    def traverse(self):
        norms = np.zeros(self.num_label)
        v = self.root
        Q, tmpQ = [v], []
        while Q:
            for u in Q:
                if u.isLeaf:
                    label, norm = u.getNorm()
                    norms[label] = norm
                else:
                    tmpQ[0:0] = u.child
            Q = tmpQ
            tmpQ = []
        return norms

    def train(self, X, Y, R):
        '''
        R: L x D, label representation
        '''
        num_sample, num_feature = X.shape
        num_sample, num_label = Y.shape
        self.num_label = num_label
        self.num_feature = num_feature

        self.X, self.Y, self.R = X, Y, R

        si = np.arange(num_sample, dtype=np.int32)
        sl = np.arange(num_label, dtype=np.int32)
        self.root = self._train(si, sl, depth=0)

        del self.X
        del self.Y
        del self.R

    def _train(self, si, sl, depth):
        num_sample = len(si)
        num_label = len(sl)

        Yt = csc_matrix(self.Y[si])[:, sl]
        v = Node()
        # meet leaf condition
        if depth >= self.max_depth or num_label <= self.num_cluster:
            print ('create leaf node: ', len(si), len(sl), depth)
            v.buildNode(self.X[si], Yt, sl, isLeaf=True, rs=self.rand_seed)
            return v

        #kmeans = KMeans(n_clusters=self.num_cluster, max_iter=3,\
        #                n_init=2, n_jobs=8, random_state=self.rand_seed).fit(self.R[sl])
        kmeans = MiniBatchKMeans(n_clusters=self.num_cluster, max_iter=3, init='random', batch_size=500,\
                                 reassignment_ratio=0, verbose=0,\
                                 max_no_improvement=1, random_state=self.rand_seed).fit(self.R[sl])
        labels = kmeans.labels_
        #eps = 1e-4
        #labels = ckmeans.kmeans(self.R[sl],\
        #                        eps, self.num_cluster, self.rand_seed)

        C = ckmeans.group_by_value(labels)
        num_cluster = len(C)
        print ('# effective clusters: ', num_cluster)

        v.child = []
        row, col, data = [], [], []
        num_active = 0
        for l in range(num_cluster):
            #cur_row = list(itertools.chain.from_iterable([Yt[:, i].indices for i in c]))
            c = np.array(C[l])
            cur_row = ckmeans.updateR(c, Yt.indices, Yt.indptr)
            if len(cur_row) == 0:
                continue
            row[0:0] = cur_row
            col[0:0] = [num_active] * len(cur_row)
            num_active += 1

            v.child.append(self._train(si[cur_row], sl[c], depth+1))
        data = [1] * len(row)
        Y = csc_matrix((data, (row, col)), shape=(num_sample, num_active),
                       dtype=np.int8)
        v.buildNode(self.X[si], Y, np.arange(num_active, dtype=np.int32), rs=self.rand_seed)

        return v

    '''
    def predict_per(self, X, topk=5):
        X = csr_matrix(hstack((X, csr_matrix(np.ones((X.shape[0], 1))))))
        num_sample, _ = X.shape
        pred, prob = [], [None] * num_sample
        for i in range(num_sample):
            _, prob[i] = self._predict(X[i])
        return pred, vstack(prob)
    '''

    def predict(self, X, topk=5):
        # append 1 to each sample
        num_sample, _ = X.shape
        beam_size = 10
        discount = 0.98

        N, I, P = [self.root], [np.arange(num_sample, dtype=np.int32)], [[0]*num_sample]

        row, col, data = [], [], []

        layer = 0
        while layer <= self.max_depth:
            print (f'start testing layer {layer}...')

            num_nodes = 0

            new_N = []
            r, c, d = [], [], []
            for v, idx, p in zip(N, I, P):
                if len(idx) == 0:
                    continue
                Yv = v.predict(X[idx]).astype(np.float64)

                #print('v.predict done...')
                if v.isLeaf == True:
                    #print('LEAF: update r, c, d starts...')
                    row[0:0], col[0:0], data[0:0] = ckmeans.updateRCD(idx, v.labels,\
                                      Yv.indices, Yv.indptr, Yv.data,\
                                      np.array(p)*discount, 1)
                    #print('LEAF: update r, c, d done...')
                    '''
                    for i in range(len(idx)):
                        row[0:0] = [idx[i]] * sz
                        col[0:0] = v.labels[Yv[i].indices]
                        data[0:0] = np.exp(Yv[i].data + p[i] * discount)
                    '''
                    continue

                '''
                for i in range(len(idx)):
                    r[0:0] = [idx[i]] * sz
                    c[0:0] = Yv[i].indices + ptr#range(ptr, ptr+sz)
                    d[0:0] = Yv[i].data + p[i] * discount
                '''

                #print('update r, c, d starts...')
                r[0:0], c[0:0], d[0:0] = ckmeans.updateRCD(idx, v.labels,\
                                    Yv.indices, Yv.indptr, Yv.data,\
                                    np.array(p)*discount, 0)
                #print('update r, c, d done...')
                num_nodes += len(v.labels)
                new_N[0:0] = reversed(v.child)

            if layer == self.max_depth:
                break
            # reverse
            N = reversed(new_N)

            Y = csr_matrix((d, (r, c)), shape=(num_sample, num_nodes),\
                           dtype=np.float64)
            #print ('build csr_matrix Y done...')

            I, P = ckmeans.updateIP(len(new_N), num_sample,\
                                   Y.indices, Y.indptr, Y.data,\
                                   beam_size)
            '''
            for i in range(num_sample):
                idx = np.argsort(Y[i].data)[-beam_size:]
                for j in idx:
                    new_nodeX[Y[i].indices[j]][1].append(i)
                    new_nodeX[Y[i].indices[j]][2].append(Y[i].data[j])
            '''
            #print ('update nodeX done...', len(I))

            layer += 1

        print ('prediction done...')
        pred = csr_matrix((data, (row, col)), shape=(num_sample, self.num_label),\
                          dtype=np.float64)
        return [], pred

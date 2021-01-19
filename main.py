from Layer import Layer
from scipy.sparse import hstack, csr_matrix, find, csc_matrix, vstack
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from evaluation import *
from sklearn.preprocessing import MultiLabelBinarizer
from eda import eda
import scipy.sparse as smat
from scipy.sparse.linalg import norm
import argparse
from xclib.data import data_utils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('TTL')
parser.add_argument('--dataset', '-d', metavar='DATASET', type=str, default='eurlex',
                                               help='choose dataset to preceed')
parser.add_argument('--num_forest', '-nf', metavar='NUM_FOREST', type=int, default=1,
                                               help='set num of forest each layer')
parser.add_argument('--num_tree', '-nt', metavar='NUM_TREE', type=int, default=3,
                                               help='set num of tree each forest')
parser.add_argument('--max_depth', '-md', metavar='MAX_DEPTH', type=int, default=1,
                                               help='set max depth of trees')
parser.add_argument('--max_layer', '-ml', metavar='MAX_LAYERE', type=int, default=2,
                                               help='set max layers')
parser.add_argument('--full_train', '-ft', metavar='FULL_TRAIN', type=int, default=0,
                                               help='choose training mode, use full training set or not')
parser.add_argument('--threshold', '-thr', metavar='THRESHOLD', type=float, default=0.3,
                                               help='choose threshold to split data into Head label and Tail label')
parser.add_argument('--augment', '-aug', metavar='AUGMENT', type=int, default=0,
                                               help='choose if augmentation for tail label will be used')
parser.add_argument('--tree_type', '-tt', metavar='Tree_TYPE', type=str, default='bonsai',
                                               choices = ['bonsai', 'craftxml', 'fastxml'],
                                               help='choose training mode, use full training set or not')
args = parser.parse_args()


def csr2list(M):
    row, col, _ = find(M)
    res = [[] for _ in range(M.shape[0])]
    for r, c in zip(row, col):
        res[r].append(c)
    return res


class Cascade:
    def __init__(self, max_layer=2, num_forest=1, n_estimator=3, max_depth=1, tree_type='fastxml', full_train=0, threshold=0.3, augment=0, weights=None):
        self.max_layer = max_layer
        self.num_forest = num_forest
        self.n_estimator = n_estimator
        self.tree_type = tree_type
        self.full_train = full_train
        self.max_depth = max_depth
        self.augment = augment
        self.weights = weights
        self.threshold = [0] + [threshold] + [1]
        self.model = []
        self.ratio = []
        self.train_label = []
        if self.full_train:
            self.max_layer = 1

    def getNorm(self):
        norms = [[] for _ in range(len(self.model))]
        for clf in self.model:
            norm = clf.getNorm()
            for i in range(len(self.model)):
                norms[i] += list(norm[i])
        return np.array(norms)

    def train(self, X, Y):
        """
        :param train_data_raw: array, whose shape is (num_samples, num_features)
        :param train_label_raw: array, whose shape is (num_samples, num_labels)
        :param supervise: string, (e.g. "hamming loss", "one-error")
        :param n_estimators: int, num_trees in each forest
        """
        Xtr = X.copy()

        num_sample, num_label = Y.shape

        for layer_index in range(self.max_layer):
            print("training layer " + str(layer_index))

            ######
            tl = int(self.threshold[layer_index] * num_label)
            tr = int(self.threshold[layer_index + 1] * num_label)
            if self.full_train:
                tl, tr = 0, num_label

            if tr - tl > 320000:
                self.max_depth = 2
            else:
                self.max_detph = 1
            layer = Layer(layer_index, self.num_forest, self.n_estimator, self.max_depth, self.tree_type, weights=self.weights)

            train_label = csr_matrix(csc_matrix(Y)[:, tl:tr])
            train_idx = []
            row, _, _ = find(train_label)
            train_idx = np.unique(row)
            train_data = Xtr[train_idx]
            train_label = train_label[train_idx]
            print (train_data.shape)
            print (train_label.shape)
            print("data preparation done...")
            ######
            if layer_index == 1:
                num_aug = self.augment
            else:
                # augment tail label only
                num_aug = 0
                # augment all labels
                #num_aug = 3

            if num_aug > 0:
                train_data, train_label = eda(train_data, train_label, num_aug=num_aug)
            print (train_data.shape)
            print (train_label.shape)
            print("data augmentation done...")

            layer.train(train_data, train_label)
            print("model training finished...")

            self.model.append(layer)

    def predict(self, X, Y=None, topk=5):
        """
        :param test_data_raw: array, whose shape is (num_test_samples, num_features)
        :return prob: array, whose shape is (num_test_samples, num_labels)
        """
        final_prob = []
        for i in range(len(self.threshold) - 1):
            print("cascade testing layer " + str(i))
            clf = self.model[i]

            res, prob = clf.predict(X)
            final_prob.append(prob)

            if self.full_train == True:
                break

        final_prob = csr_matrix(hstack(final_prob))
        res = np.zeros((X.shape[0], topk))
        for i in range(X.shape[0]):
            y = np.argsort(final_prob[i].data)[-topk:][::-1]
            res[i, :len(y)] = final_prob[i].indices[y]

        return res, final_prob

#dataset = 'bibtex'
#dataset = 'eurlex'
#dataset = 'Eurlex'
#dataset = 'Wiki10-31K'
dataset = args.dataset
files = ['X.trn.npz', 'Y.trn.npz', 'X.val.npz', 'Y.val.npz', 'X.tst.npz', 'Y.tst.npz']

data = []
for file in files:
    d = smat.load_npz('./datasets/' + dataset + '/' + file)
    data.append(d)

Xtr, Ytr, Xva, Yva, Xte, Yte = data

Xtr = vstack([Xtr, Xva])
Ytr = vstack([Ytr, Yva])

#####
num_sample, num_label = Ytr.shape
label_count = np.zeros(num_label)
_, col, _ = find(Ytr)
for c in col:
    label_count[c] += 1

test_label_count = np.zeros(num_label)
_, col, _ = find(Yte)
for c in col:
    test_label_count[c] += 1

### sort labels in decending order
label_count_idx = np.argsort(label_count)[::-1]

np.savetxt(dataset + '_label_count.txt', label_count[label_count_idx])
X_embedded = TSNE(n_components=2).fit_transform(Xtr.toarray())
plt.plot(X_embedded)

#print (label_count[label_count_idx])
Ytr = csr_matrix(csc_matrix(Ytr)[:, label_count_idx])
Yte = csr_matrix(csc_matrix(Yte)[:, label_count_idx])

### feature normalization
row_sums = norm(Xtr, axis=1)
row_indices, _ = Xtr.nonzero()
Xtr.data /= row_sums[row_indices]

row_sums = norm(Xte, axis=1)
row_indices, _ = Xte.nonzero()
Xte.data /= row_sums[row_indices]

eps = 0
if args.dataset.startswith('DeliciousLa'):
    eps = 0.1
if args.dataset.startswith('Amazon-670K'):
    eps = 0.1
Xtr.data[Xtr.data < eps] = 0
Xtr.eliminate_zeros()

Xte.data[Xte.data < eps] = 0
Xte.eliminate_zeros()

print (Xtr.shape)
print (Ytr.shape)
print (Xte.shape)
print (Yte.shape)
#print (label_count[label_count_idx])
#print (','.join(map(str, label_count[label_count_idx])))
#print (','.join(map(str, test_label_count[label_count_idx])))

train_labels = csr2list(Ytr)
if args.dataset.startswith('Wikipedia'):
    a, b = 0.5, 0.4
elif args.dataset.startswith('Amazon-670K'):
    a, b = 0.6, 2.6
else:
    a, b = 0.55, 1.5

mlb = MultiLabelBinarizer(range(Yte.shape[1]), sparse_output=True)
targets = mlb.fit_transform(csr2list(\))
inv_w = get_inv_propensity(mlb.transform(train_labels), a, b)

clf = Cascade(max_layer=args.max_layer, num_forest=args.num_forest, n_estimator=args.num_tree,\
              max_depth=args.max_depth, tree_type=args.tree_type, full_train=args.full_train,\
              threshold=args.threshold, augment=args.augment,\
              weights=inv_w)
clf.train(Xtr, Ytr)

res, prob = clf.predict(Xte, Yte)
norms = clf.getNorm()
norms = np.mean(norms, axis=0)


print(f'Precision@1,3,5: {get_p_1(res, targets, mlb)}, {get_p_3(res, targets, mlb)}, {get_p_5(res, targets, mlb)}')
print(f'nDCG@1,3,5: {get_n_1(res, targets, mlb)}, {get_n_3(res, targets, mlb)}, {get_n_5(res, targets, mlb)}')
print('PSPrecision@1,3,5:', get_psp_1(res, targets, inv_w, mlb), get_psp_3(res, targets, inv_w, mlb), get_psp_5(res, targets, inv_w, mlb))
print('PSnDCG@1,3,5:', get_psndcg_1(res, targets, inv_w, mlb), get_psndcg_3(res, targets, inv_w, mlb), get_psndcg_5(res, targets, inv_w, mlb))

if args.full_train:
    if args.augment > 0:
        path = './aug_results/' + dataset + '_fulltrain'
        np.savetxt(dataset+'_norms_aug', norms)
    else:
        path = './results/' + dataset + '_fulltrain'
        np.savetxt(dataset+'_norms', norms)
else:
    if args.augment > 0:
        path = './aug_results/' + dataset + '_' + str(args.augment) + '_' + str(args.threshold)
        np.savetxt(dataset+'_norms_aug_' + str(args.augment) + '_' + str(args.threshold), norms)
    else:
        path = './results/' + dataset + '_' + str(args.threshold)
        np.savetxt(dataset+'_norms_' + str(args.threshold), norms)


results = open(path, 'w')

results.write(f'Precision@1,3,5: {get_p_1(res, targets, mlb)}, {get_p_3(res, targets, mlb)}, {get_p_5(res, targets, mlb)}')
results.write('\n')
results.write(f'nDCG@1,3,5: {get_n_1(res, targets, mlb)}, {get_n_3(res, targets, mlb)}, {get_n_5(res, targets, mlb)}')
results.write('\n')
results.write(f'PSPrecision@1,3,5: {get_psp_1(res, targets, inv_w, mlb)}, {get_psp_3(res, targets, inv_w, mlb)}, {get_psp_5(res, targets, inv_w, mlb)}')
results.write('\n')
results.write(f'PSnDCG@1,3,5: {get_psndcg_1(res, targets, inv_w, mlb)}, {get_psndcg_3(res, targets, inv_w, mlb)}, {get_psndcg_5(res, targets, inv_w, mlb)}')
results.write('\n')

topk=5
num_sample = prob.shape[0]
for i in range(num_sample):
    y = np.argsort(prob[i].data * inv_w[prob[i].indices])[-topk:][::-1]
    res[i, :len(y)] = prob[i].indices[y]

print ('======with re-ranking====')
print(f'Precision@1,3,5: {get_p_1(res, targets, mlb)}, {get_p_3(res, targets, mlb)}, {get_p_5(res, targets, mlb)}')
print(f'nDCG@1,3,5: {get_n_1(res, targets, mlb)}, {get_n_3(res, targets, mlb)}, {get_n_5(res, targets, mlb)}')
print('PSPrecision@1,3,5:', get_psp_1(res, targets, inv_w, mlb), get_psp_3(res, targets, inv_w, mlb), get_psp_5(res, targets, inv_w, mlb))
print('PSnDCG@1,3,5:', get_psndcg_1(res, targets, inv_w, mlb), get_psndcg_3(res, targets, inv_w, mlb), get_psndcg_5(res, targets, inv_w, mlb))

results.write('======with re-ranking=====\n')
results.write(f'Precision@1,3,5: {get_p_1(res, targets, mlb)}, {get_p_3(res, targets, mlb)}, {get_p_5(res, targets, mlb)}')
results.write('\n')
results.write(f'nDCG@1,3,5: {get_n_1(res, targets, mlb)}, {get_n_3(res, targets, mlb)}, {get_n_5(res, targets, mlb)}')
results.write('\n')
results.write(f'PSPrecision@1,3,5: {get_psp_1(res, targets, inv_w, mlb)}, {get_psp_3(res, targets, inv_w, mlb)}, {get_psp_5(res, targets, inv_w, mlb)}')
results.write('\n')
results.write(f'PSnDCG@1,3,5: {get_psndcg_1(res, targets, inv_w, mlb)}, {get_psndcg_3(res, targets, inv_w, mlb)}, {get_psndcg_5(res, targets, inv_w, mlb)}')
results.write('\n')

results.flush()
results.close()

#prob = prob * 10**3
#prob = prob.astype(np.int32)
#prob.eliminate_zeros()
#data_utils.write_sparse_file(prob, path+"_scores.txt")

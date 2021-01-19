
import random
from random import shuffle
from kmeans import random_deletion as crandom_deletion
from kmeans import random_swap as crandom_swap
from scipy.sparse import csr_matrix, vstack
random.seed(1)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    #obviously, if there's only one word, don't delete it
    if words.nnz == 1:
        return words

    #randomly delete words with probability p
    new_words = words.copy()
    for i in range(words.nnz):
        r = random.uniform(0, 1)
        if r < p:
            new_words.data[i] = 0

    #if you end up deleting all words, just return a random word
    if new_words.count_nonzero == 0:
        rand_int = random.randint(0, words.nnz-1)
        new_words.data[rand_int] = words.data[rand_int]
    new_words.eliminate_zeros()

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(words):
    if words.nnz == 0:
        return words
    idx_1 = random.randint(0, words.nnz-1)
    idx_2 = idx_1
    counter = 0
    while idx_2 == idx_1:
        idx_2 = random.randint(0, words.nnz-1)
        counter += 1
        if counter > 3:
            return words
    words.data[idx_1], words.data[idx_2] = words.data[idx_2], words.data[idx_1]
    return words

########################################################################
# main data augmentation function
########################################################################

def eda(words, labels=None, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=2):
    # words is type of csr_matrix with length equaling to 1
    # This function returns a list of csr_matrix with length 1

    num_words = words.nnz // words.shape[0]

    augmented_sentences = []
    augmented_labels = []
    num_new_per_technique = int(num_aug/2)+1
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    #rs
    for _ in range(num_new_per_technique):
        ind, indptr, data = crandom_swap(words.indices, words.indptr, words.data, n_rs)
        a_words = csr_matrix((data, ind, indptr), shape=words.shape)
        a_words.eliminate_zeros()
        #a_words = random_swap(words, n_rs)
        augmented_sentences.append(a_words)
        augmented_labels.append(labels)

    #rd
    for _ in range(num_new_per_technique):
        ind, indptr, data = crandom_deletion(words.indices, words.indptr, words.data, p_rd)
        a_words = csr_matrix((data, ind, indptr), shape=words.shape)
        a_words.eliminate_zeros()
        #ind, data = crandom_deletion(words.indices, words.data, p_rd)
        #a_words = csr_matrix((data, ind, [0, len(data)]), shape=words.shape)
        #a_words = random_deletion(words, p_rd)
        augmented_sentences.append(a_words)
        augmented_labels.append(labels)

    #random.seed(1234)
    #shuffle(augmented_sentences)
    #random.seed(1234)
    #shuffle(augmented_labels)

    #trim so that we have the desired number of augmented sentences
    '''
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        random.seed(1234)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
        random.seed(1234)
        augmented_labels = [s for s in augmented_labels if random.uniform(0, 1) < keep_prob]
    '''

    #append the original sentence
    augmented_sentences.append(words)
    augmented_labels.append(labels)

    return vstack(augmented_sentences), vstack(augmented_labels)

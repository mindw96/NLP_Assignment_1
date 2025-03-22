import numpy as np
from scipy import stats
import pandas as pd
import random
import os
import warnings
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def data_loading(path):
    data_list = []
    with open(path, mode='r', encoding='UTF8') as f:
        temp = f.readlines()
        for item in temp:
            data_list.append(item.split('\n')[0])

    return data_list


def preprocess(vocab):
    word_to_id = {}
    id_to_word = {}

    for word in vocab:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    return word_to_id, id_to_word


def make_co_matrix(data=None, vocab_1=None, vocab_2=None, window_size=1):
    vocab_1_word_to_id, vocab_1_id_to_word = preprocess(vocab_1)
    vocab_2_word_to_id, vocab_2_id_to_word = preprocess(vocab_2)

    matrix = np.zeros((len(vocab_1), len(vocab_2)), dtype=np.int32)

    for text in data:
        tokens = text.split()
        padded_tokens = ['<s>'] * window_size + tokens + ['</s>'] * window_size

        for center_pos in range(window_size, len(padded_tokens) - window_size):
            center_word = padded_tokens[center_pos]

            if center_word in vocab_1_word_to_id:
                center_id = vocab_1_word_to_id[center_word]

                # Check context words in the window
                for context_pos in range(center_pos - window_size, center_pos + window_size + 1):
                    # Skip the center word itself
                    if context_pos == center_pos:
                        continue

                    context_word = padded_tokens[context_pos]

                    # Check if context word is in vocab_2 (column vocabulary)
                    if context_word in vocab_2_word_to_id:
                        context_id = vocab_2_word_to_id[context_word]
                        matrix[center_id, context_id] += 1

    return matrix


def cos_similarity(x, y, eps=1e-8):
    return np.dot(x, y) / (np.multiply(np.sqrt(np.sum(x ** 2)), np.sqrt(np.sum(y ** 2))) + eps)


def evaluate(data_path, matrix, vocab):
    word_to_id, id_to_word = preprocess(vocab)
    data = pd.read_table(data_path)

    sim_list = []
    score_list = []
    for idx in range(len(data)):
        temp = data.loc[idx]
        word1 = temp[0]
        word2 = temp[1]
        score = temp[2]
        score_list.append(score)

        word1_idx = word_to_id[word1]
        word2_idx = word_to_id[word2]
        cos_sim = cos_similarity(matrix[word1_idx], matrix[word2_idx])

        if np.isnan(cos_sim):
            cos_sim = 0.0
        sim_list.append(cos_sim)

    spear_score = stats.spearmanr(score_list, sim_list)

    return spear_score


def make_pmi(matrix, verbose=True):
    eps = 1e-10
    pmi = np.zeros_like(matrix, dtype=np.float32)

    row = np.sum(matrix, axis=1)
    col = np.sum(matrix, axis=0)
    total_sum = np.sum(matrix)

    total = matrix.shape[0] * matrix.shape[1]
    cnt = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                pmi[i, j] = np.log((matrix[i, j] * total_sum) / (col[j] * row[i] + eps))
                # pmi[i, j] = max(0, np.log((matrix[i, j] * total_sum) / (col[j] * row[i] + eps)))

            if verbose:
                cnt += 1
                if cnt % (total // 10) == 0:
                    print('{}% '.format(int(100 * cnt / total)), end='')

    return pmi


def nearest_neighbor(words=None, vocab=None, pmi_1=None, pmi_6=None):
    word_to_id, id_to_word = preprocess(vocab)

    for word in words:
        sim_1_dic = {}
        sim_6_dic = {}

        for idx in range(len(vocab)):
            if id_to_word[idx] == word:
                continue
            else:
                sim_1 = cos_similarity(pmi_1[word_to_id[word]], pmi_1[idx])
                sim_1_dic[id_to_word[idx]] = sim_1

                sim_6 = cos_similarity(pmi_6[word_to_id[word]], pmi_6[idx])
                sim_6_dic[id_to_word[idx]] = sim_6

        sim_1_sorted = sorted(sim_1_dic.items(), key=lambda x: x[1], reverse=True)
        sim_6_sorted = sorted(sim_6_dic.items(), key=lambda x: x[1], reverse=True)

        print('-' * 80)
        print('"{}" Similarity Top 10'.format(word))
        print('Window size 1, 6')
        for i in range(10):
            print('{:.4f} {:15} {:.4f} {:15}'.format(sim_1_sorted[i][1], sim_1_sorted[i][0],
                                                     sim_6_sorted[i][1], sim_6_sorted[i][0]))

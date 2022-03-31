import numpy as np
from scipy import stats
import pandas as pd
import random
import os
import warnings

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
        texts = text.split(' ')
        texts.insert(0, '<s>')
        texts.append('</s>')

        for idx, center_word in enumerate(texts):
            if center_word in vocab_1_word_to_id and center_word in vocab_2_word_to_id:
                matrix[vocab_1_word_to_id[center_word], vocab_2_word_to_id[center_word]] += 1

            for i in range(1, window_size + 1):
                left_idx = idx - i
                right_idx = idx + i
                if center_word in vocab_1_word_to_id:
                    word_id = vocab_1_word_to_id[center_word]

                    if left_idx >= 0:
                        if texts[left_idx] in vocab_2_word_to_id:
                            left_word_id = vocab_2_word_to_id[texts[left_idx]]
                            matrix[word_id, left_word_id] += 1

                    if right_idx < len(texts):
                        if texts[right_idx] in vocab_2_word_to_id:
                            right_word_id = vocab_2_word_to_id[texts[right_idx]]
                            matrix[word_id, right_word_id] += 1

    return matrix


def cos_similarity(x, y, eps=1e-8):
    return np.dot(x, y) / (np.multiply(np.sqrt(np.sum(x ** 2)), np.sqrt(np.sum(y ** 2))) + eps)


def evaluate(data_path, matrix, vocab):
    word_to_id, id_to_word = preprocess(vocab)
    data = pd.read_table(data_path)

    spear_scores = []
    for idx in range(len(data)):
        temp = data.loc[idx]
        word1 = temp[0]
        word2 = temp[1]
        score = temp[2]

        word1_idx = word_to_id[word1]
        word2_idx = word_to_id[word2]
        cos_sim = cos_similarity(matrix[word1_idx], matrix[word2_idx])

        if np.isnan(cos_sim):
            cos_sim = 0.0

        spear_score = stats.spearmanr(score, cos_sim)
        spear_scores.append(spear_score)

    return spear_scores


def make_pmi(matrix, verbose=True):
    eps = 1e-8
    pmi = np.zeros_like(matrix, dtype=np.float32)

    row = []
    for idx in range(matrix.shape[0]):
        row.append(sum(matrix[idx]))

    col = []
    for idx in range(matrix.shape[1]):
        col.append(sum(matrix[:, idx]))

    total_sum = sum(row)

    total = matrix.shape[0] * matrix.shape[1]
    cnt = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pmi[i, j] = max(0, np.log((matrix[i, j] * total_sum) / (col[j] * row[i]) + eps))

            if verbose:
                cnt += 1
                if cnt % (total // 10) == 0:
                    print('{}% 완료'.format(100 * cnt / total))

    return pmi


def nearest_neighbor(words=None, vocab=None, pmi_1=None, pmi_6=None):
    word_to_id, id_to_word = preprocess(vocab)

    for word in words:
        sim_list_1 = []
        sim_list_6 = []

        for idx in range(len(vocab)):
            if id_to_word[idx] == word:
                continue
            sim_1 = cos_similarity(pmi_1[word_to_id[word]], pmi_1[idx])
            sim_list_1.append(sim_1)

            sim_6 = cos_similarity(pmi_6[word_to_id[word]], pmi_6[idx])
            sim_list_6.append(sim_6)

        print('-' * 80)
        print('"{}" Similarity Top 10'.format(word))
        print('Window size 1, 6')
        for _ in range(10):
            sim_1_max = max(sim_list_1)
            index_1 = sim_list_1.index(sim_1_max)

            sim_6_max = max(sim_list_6)
            index_6 = sim_list_6.index(sim_6_max)

            print('{:.4f} {:15} {:.4f} {:15}'.format(sim_1_max, id_to_word[index_1], sim_6_max,
                                                                    id_to_word[index_6]))
            sim_list_1.remove(sim_1_max)
            sim_list_6.remove(sim_6_max)


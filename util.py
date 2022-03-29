import numpy as np
from scipy import stats
import pandas as pd
import random
import os


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


def preprocess(text_list):
    word_to_id = {}
    id_to_word = {}

    for word in text_list:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    return word_to_id, id_to_word


def make_corpus(text):
    words = text.split(' ')
    word_to_id = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id

    corpus = np.array([word_to_id[w] for w in words])

    return corpus


def make_co_matrix(data=None, vocab_1=None, vocab_2=None, window_size=1):
    vocab_1_word_to_id, vocab_1_id_to_word = preprocess(vocab_1)
    vocab_2_word_to_id, vocab_2_id_to_word = preprocess(vocab_2)

    co_matrix = np.zeros((len(vocab_1), len(vocab_2)), dtype=np.int32)

    for text in data:
        texts = text.split(' ')
        texts.insert(0, '<s>')
        texts.append('</s>')
        for idx, center_word in enumerate(texts):
            if center_word in vocab_1_word_to_id and center_word in vocab_2_word_to_id:
                co_matrix[vocab_1_word_to_id[center_word], vocab_2_word_to_id[center_word]] += 1

            for i in range(1, window_size + 1):
                left_idx = idx - i
                right_idx = idx + i
                if center_word in vocab_1_word_to_id:
                    word_id = vocab_1_word_to_id[center_word]

                    if left_idx >= 0:
                        if texts[left_idx] in vocab_2_word_to_id:
                            left_word_id = vocab_2_word_to_id[texts[left_idx]]
                            co_matrix[word_id, left_word_id] += 1
                        else:
                            continue

                    if right_idx < len(texts):
                        if texts[right_idx] in vocab_2_word_to_id:
                            right_word_id = vocab_2_word_to_id[texts[right_idx]]
                            co_matrix[word_id, right_word_id] += 1
                        else:
                            continue
                else:
                    continue

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    return np.dot(x, y) / (np.multiply(np.sqrt(np.sum(x ** 2)), np.sqrt(np.sum(y ** 2))) + eps)


def evaluate(data_path, matrix, vocab):
    word_to_id, id_to_word = preprocess(vocab)
    data = pd.read_table(data_path)
    scores = []
    sim_scores = []

    for idx in range(len(data)):
        temp = data.loc[idx]
        word1 = temp[0]
        word2 = temp[1]
        score = temp[2]
        scores.append(score)
        word1_idx = word_to_id[word1]
        word2_idx = word_to_id[word2]
        cos_sim = cos_similarity(matrix[word1_idx], matrix[word2_idx])
        sim_scores.append(cos_sim)

    spear_score = stats.spearmanr(scores, sim_scores)

    return spear_score


def make_pmi(matrix, verbose=True):
    eps = 1e-8
    pmi = np.zeros_like(matrix, dtype=np.float32)

    N = []
    for idx in range(matrix.shape[0]):
        N.append(sum(matrix[idx]))

    S = []
    for idx in range(matrix.shape[1]):
        S.append(sum(matrix[:, idx]))

    total_sum = sum(N)
    print(total_sum)

    total = matrix.shape[0] * matrix.shape[1]
    cnt = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pmi[i, j] = max(0, (matrix[i, j] * total_sum) / (S[j] * N[i]) + eps)
            cnt += 1
            if verbose:
                if cnt % (total // 10) == 0:
                    print('{}% 완료'.format(100 * cnt / total))

    return pmi

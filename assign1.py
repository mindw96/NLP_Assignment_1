import numpy as np

V_path = './data/vocab-wordsim.txt'
VC_path = './data/vocab-25k.txt'


def data_preprocessing(path):
    data_list = []
    with open(path, mode='r', encoding='UTF8') as f:
        temp = f.readlines()
        for item in temp:
            data_list.append(item.split('\n')[0])

    return data_list


V = data_preprocessing(V_path)
VC = data_preprocessing(VC_path)

print(len(V))
print(len(VC))

window_size = 3

corpus_size = len(V)
co_matrix = np.zeros((len(V), len(VC)), dtype=np.int32)

for idx, word in enumerate(V):
    for i in range(1, window_size + 1):
        word_id = V.index(word)
        left_idx = idx - i
        right_idx = idx + i

        if left_idx >= 0:
            left_word = VC[left_idx]
            print(word, left_word)
            if word == left_word:
                print(word, left_word)
                left_word_id = VC.index(left_word)
                co_matrix[word_id, left_word_id] += 1

        if right_idx < corpus_size:
            right_word = VC[right_idx]
            print(word, right_word)
            if word == right_word:
                print(word, right_word)
                right_word_id = VC.index(right_word)
                co_matrix[word_id, right_word_id] += 1

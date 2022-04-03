from util import seed_everything, data_loading, make_co_matrix, make_pmi, nearest_neighbor


def main():
    print('=' * 80)
    print('Assignment 1.4')

    seed_everything(9608)

    VC_path = './data/vocab-25k.txt'
    wiki_1_path = './data/wiki-1percent.txt'

    VC = data_loading(VC_path)
    wiki_1 = data_loading(wiki_1_path)

    co_matrix_1 = make_co_matrix(wiki_1, VC, VC, window_size=1)
    co_matrix_6 = make_co_matrix(wiki_1, VC, VC, window_size=6)

    pmi_1 = make_pmi(co_matrix_1, verbose=True)
    pmi_6 = make_pmi(co_matrix_6, verbose=True)

    print('\nAssign 1.4.1')
    nearest_neighbor(words=['monster'], vocab=VC, pmi_1=pmi_1, pmi_6=pmi_6)

    print('\nAssignment 1.4.2')
    words = ['keyboards', 'canceled', 'worried', 'about']
    nearest_neighbor(words=words, vocab=VC, pmi_1=pmi_1, pmi_6=pmi_6)

    print('\nAssignment 1.4.3')
    words = ['bank', 'cell', 'apple', 'apples', 'axes', 'frame', 'light', 'well']
    nearest_neighbor(words=words, vocab=VC, pmi_1=pmi_1, pmi_6=pmi_6)


if __name__ == '__main__':
    main()

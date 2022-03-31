from util import seed_everything, data_loading, evaluate, make_co_matrix


def main():
    seed_everything(9608)

    V_path = './data/vocab-wordsim.txt'
    VC_path = './data/vocab-25k.txt'
    wiki_1_path = './data/wiki-1percent.txt'
    men_path = './data/men.txt'
    simlex_path = './data/simlex-999.txt'

    V = data_loading(V_path)
    VC = data_loading(VC_path)
    wiki_1 = data_loading(wiki_1_path)

    co_matrix = make_co_matrix(wiki_1, V, VC, window_size=6)

    spear_score_men = evaluate(men_path, co_matrix, V)
    print(spear_score_men)

    spear_score_simlex = evaluate(simlex_path, co_matrix, V)
    print(spear_score_simlex)


if __name__ == '__main__':
    main()

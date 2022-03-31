from util import seed_everything, data_loading, make_co_matrix, make_pmi, evaluate


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

    for window_size in [1, 3, 6]:
        print('-' * 80)
        print('Window Size: {}'.format(window_size))

        co_matrix = make_co_matrix(wiki_1, V, VC, window_size=window_size)
        pmi = make_pmi(co_matrix, verbose=False)

        c_spear_men = evaluate(men_path, co_matrix, V)
        print('C Spearman Score compare with MEN: {:.4f}'.format(c_spear_men[0]))

        pmi_spear_men = evaluate(men_path, pmi, V)
        print('PMI Spearman Score compare with MEN: {:.4f}'.format(pmi_spear_men[0]))

        c_spear_simlex = evaluate(simlex_path, co_matrix, V)
        print('C Spearman Score compare with Simlex999: {:.4f}'.format(c_spear_simlex[0]))

        pmi_spear_simlex = evaluate(simlex_path, pmi, V)
        print('PMI Spearman Score compare with Simlex999: {:.4f}'.format(pmi_spear_simlex[0]))


if __name__ == '__main__':
    main()

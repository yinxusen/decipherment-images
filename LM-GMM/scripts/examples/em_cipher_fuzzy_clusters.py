import sys

import numpy as np

from decipher.lm_c_gmm import em_restart_fix_gmm, viterbi
from decipher.utils import eprint, log_p_e, levenshtein


if __name__ == '__main__':

    fuzzy_clusters = np.load(sys.argv[1])['link_tbl']
    k, n = fuzzy_clusters.shape

    bigram = np.load(sys.argv[2])
    bigram_tbl = np.log(bigram['prb_tbl'])
    unigram_tbl = np.log(bigram['sum_tbl'])
    alphabet = bigram['alphabet']

    gold = \
        'comewritersandcriticswhoprophesi' \
        'zewithyourpenandkeepfoureyeswide' \
        'thechancewontcomeagainanddontspe' \
        'aktoosoonforthewheelsstillinspin' \
        'andtheresnotellingwhothatitsnami' \
        'ngforthelosernowwillbelatertowin' \
        'forthetimestheyareachangingcomes' \
        'enatorscongressmenpleaseheedthec' \
        'alldontstandinthedoorwaydontbloc' \
        'kupthehallforhethatgetshurtwillb' \
        'ehewhohasstalledtheresabattleout' \
        'sideanditisratingitllsoonshakeyo' \
        'urwindowsandbattlefourwallsforth' \
        'etimestheyareachangingcomemother' \
        'sandfathersthroughoutthelandandd' \
        'ontcriticizewhatyoucantunderstan' \
        'dyoursonsandyourdaughtersarebeyo' \
        'ndyourcommandyouroldroadisrapidl' \
        'yagingpleasegetoutofthenewoneify' \
        'oucanblendyourhandforthetimesthe' \
        'yareachanging'

    def subst_tbl_init_func():
        subst_init_tbl = np.random.random((k, k))
        subst_init_tbl /= np.sum(subst_init_tbl, axis=1)[:, np.newaxis]
        subst_init_tbl = np.log(subst_init_tbl)
        return subst_init_tbl

    restart = 50

    for i in range(restart):
        eprint('trial {}'.format(i))
        subst_tbl, xe, pc = em_restart_fix_gmm(
            unigram_tbl, bigram_tbl, fuzzy_clusters, subst_tbl_init_func,
            restart=0)
        score, e_path = viterbi(
            list(range(n)), unigram_tbl, bigram_tbl,
            fuzzy_clusters, 3 * subst_tbl)
        ll_lm = log_p_e(e_path, unigram_tbl, bigram_tbl)
        symbols = ''.join(map(lambda i: alphabet[i], e_path))
        ed_ds = levenshtein(symbols, gold)
        eprint('decoded: \n{}'.format(symbols))
        eprint('csv: random,{},{},{},{},{}'.format(
            symbols[:10], pc, score, ll_lm, ed_ds))

import sys

import numpy as np

from decipher.lm_gmm import LMGMM
from decipher.utils import eprint, viterbi, log_p_e, levenshtein


if __name__ == '__main__':

    data_path = sys.argv[1]
    bigram_path = sys.argv[2]

    features = np.load(data_path)['features']
    k = 22
    bigram = np.load(bigram_path)
    bigram_tbl = np.log(bigram['prb_tbl'])
    unigram_tbl = np.log(bigram['sum_tbl'])
    alphabet = bigram['alphabet']

    # gold of courier
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

    init_file = 'decipher_text-arial-auto-seg.txt'
    with open(init_file, 'r') as f_init:
        eprint('open the best init file: {}'.format(init_file))
        s = f_init.readlines()[1].strip()  # load deciphered text from line 2
        eprint('read best deciphered result: \n{}'.format(s))

    init_2stage = s

    restarts = 50
    for i in range(restarts):
        eprint('trial {}'.format(i))
        cc = LMGMM(features, k,
                   3 * unigram_tbl, 3 * bigram_tbl, alphabet,
                   params_init='2stage', init_2stage=init_2stage)
        cc.fit()

        score, path = viterbi(
            list(range(len(features))), unigram_tbl, bigram_tbl, cc.link_tbl)
        decoded = ''.join([alphabet[i] for i in path])
        ll_lm = log_p_e(path, unigram_tbl, bigram_tbl)
        ed_ds = levenshtein(decoded, gold)
        eprint('decoded: \n{}'.format(decoded))
        eprint('csv: random,{},{},{},{},{}'.format(
            decoded[:10], cc.ll, score, ll_lm, ed_ds))

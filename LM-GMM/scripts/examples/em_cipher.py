import sys

import numpy as np

from decipher.lm_subst import em_decipher
from decipher.utils import viterbi, eprint, log_p_e, levenshtein

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = map(lambda s: s.strip().split(), f.readlines())

    line = lines[0]

    bigram = np.load(sys.argv[2])

    bigram_tbl = np.log(bigram['prb_tbl'])
    unigram_tbl = np.log(bigram['sum_tbl'])

    en_alphabet = bigram['alphabet']
    e_lookup = dict(map(lambda (i, e): (e, i), enumerate(en_alphabet)))
    cf_alphabet = np.unique(np.asarray(line))
    c_lookup = dict(map(lambda (i, j): (j, i), enumerate(cf_alphabet)))

    line = map(lambda e: c_lookup[e], line)

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

    restart = 50

    for i in range(restart):
        eprint('trial {}'.format(i))
        link_tbl, xe, pc = em_decipher(line, unigram_tbl, bigram_tbl)
        score, path = viterbi(line, unigram_tbl, bigram_tbl, 3 * link_tbl)
        ll_lm = log_p_e(path, unigram_tbl, bigram_tbl)
        symbols = ''.join(map(lambda i: en_alphabet[i], path))
        ed_ds = levenshtein(symbols, gold)
        eprint('decoded: \n{}'.format(symbols))
        eprint('csv: random,{},{},{},{},{}'.format(symbols[:10], pc, score, ll_lm, ed_ds))

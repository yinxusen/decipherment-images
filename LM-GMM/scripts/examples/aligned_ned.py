import sys

from decipher.aligned_ned import get_ned


if __name__ == '__main__':
    gold =\
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

    print('len of gold: {}'.format(len(gold)))

    with open(sys.argv[1], 'r') as fin:
        lines = [s.strip() for s in fin.readlines()]

    i = 0
    while i < len(lines):
        print('len of sys: {}'.format(len(lines[i].split())))
        print(lines[i])
        ned = get_ned(gold, lines[i].split())
        print('csv: {}'.format(ned))
        i += 1

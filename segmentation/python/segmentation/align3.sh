#!/bin/bash
#
#######################################################
# Usage:
#
#   % align3 <data> <mean> <sigma1> <sigma2> <p>
#
# Where <data> is a two-line file like this:
#
#   c c c
#   c29 c28 c14 c29 c13 c12 c3 c8 c30 c15 c5 c20 c22 c24 c28 c0
#
# The first line is now ignored.
#
# The second line gives the number of black pixels per slice (row or column) 
# in the image.
#
#   Suggested <mean>   = ??  # number of entities you imagine
#   Suggested <sigma1> = 4   # stdev over number of entities
#   Suggested <sigma2> = 4   # stdev over entity sizes (this mean is determined by |COLUMNS|/mean)
#   Suggested <p>      = 0.9 # penalty for segmenting across black
#
# The output lists proposed break points (slices) after each entity, eg:
#
#   7 11 16

CARMEL="`which carmel`"
FWDIR="$(cd `dirname $0`; pwd)"
pushd $FWDIR > /dev/null

R=`head -2 $1 | tail -1 | tr -d 'c' | tr ' ' '\012' | sort -nr | head -1`      # max(black per slice)

MEAN=$2              # [1..inf]  expected number of entities
SIGMA1=$3            # [1..inf]  bigger = more variation in # of entities
SIGMA2=$4            # [1..inf]  bigger = more variation in entity size
P=$5                 # [0..1]    closer to 1.0 = more penalty for cutting through black

N=`head -2 $1 | tail -1 | wc -w`     # slices 

./make-fsa2.sh $MEAN $SIGMA1 > $1.fsa
./make-fst2.sh $MEAN $N $R $SIGMA2 $P > $1.fst-p

cat $1 | head -2 | tail -1 | 
$CARMEL -sriIWk 1 $1.fsa $1.fst-p |
sed 's/ \*e\*/X/g' | tr -d 'c' | grep 'X' |
awk '{a=0; for (i=1; i<=NF; i++) {a += length($i); printf("%d ",a)} printf("\n")}' 

popd > /dev/null

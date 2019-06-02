#!/bin/bash
#
##############################################################
# Usage:
#
#   % make-fst <entities> <slices> <other-slices> <sigma> <p> 
#
# Create a WFST that converts generic entities into integer sequences.
# Each element of the sequence tells how many black pixels are in the pixel slice.
#

MAX=3   # entity size must be below MAX * expected-mean

echo '' |
awk 'BEGIN  \
     {c='$1'; n='$2'; r='$3'; sigma='$4'; p='$5';
      printf("F\n");
      for (i=1; i<=('$MAX'*(n-c)/c); i++) 
        printf("(F (%d c *e* %f))\n", i, 1);
	  for (i=1; i<=('$MAX'*(n-c)/c); i++)
        for (j=0; j<=r; j++)
	      printf("(%d (%d *e* c%d %f))\n", i, i-1, j, 1);
      for (i=0; i<=r; i++)
        printf("(0 (F *e* c%d %4.5e))\n", i, (i/r))}' |
grep -v ' 0.000000))' |
grep -v 'e-3..))'  # quick fix, just remove the line of probability


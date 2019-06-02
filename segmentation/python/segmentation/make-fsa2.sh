#!/bin/bash
#
#############################################################
#
# % make-fsa2 <mean> <stdev>
#
# Builds acceptor that eats up roughly <mean> number of c's.
# 

echo '' |
awk 'BEGIN \
     {printf("1\n");
      mean = '$1';
      stdev = '$2';
      for (i=1;i<=2*mean;i++)
        printf("(F (%d c %f))\n", i, 1 / (stdev * sqrt(2*3.1415)) * \
	                             exp(- ((i-mean) * (i-mean)) / \
				     (2 * stdev * stdev)));
      for (i=2*mean;i>=2;i--)
        printf("(%d (%d c))\n",i,i-1)}'


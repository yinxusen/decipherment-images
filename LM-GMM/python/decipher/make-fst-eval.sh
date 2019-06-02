# Copyright 2019 Xusen Yin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash
#
##############################################################
# Usage:
#
#   % make-fst <#clusters> <#golds>
#
# Create a WFST that converts generic entities into integer sequences.
# Each element of the sequence tells how many black pixels are in the pixel slice.
#

echo '' |
awk 'BEGIN  \
     {c='$1'; g='$2';
      printf("0\n");
      printf("(0 (insert *e* *e*))\n");
      printf("(0 (delete *e* *e*))\n");
      printf("(0 (subst *e* *e*))\n");
      for (i=0; i<c; i++)
          printf("(insert (0 c%d *e*))\n", i);
      for (i=0; i<g; i++)
          printf("(delete (0 *e* g%d))\n", i);
      for (i=0; i<c; i++)
          for (j=0; j<g; j++)
              printf("(subst (0 c%d g%d))\n", i, j)}'


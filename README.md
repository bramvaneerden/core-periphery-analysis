# core-periphery-analysis
Core periphery analysis project for the SNACS course

The analysis folder contains the python files we used for analysis, as well as the Json files that were created with the code.


## how to get the code for the hcp algorithm working

The HCP folder contains the algorithm proposed by Polanco et al. with a slight alteration, as we were unable to execute in the form it is on their github right now : https://github.com/apolanco115/hcp

Before doing any make commands, in the CMakeLists.txt file:

1. remove [mvector.cpp and mvector.h] from the list in the add_executable line

2. [add indexed_list.cpp,  indexed_list.h AND node.h] to add_executable line

Also note: the algorithm only seemed to work for networks up to 32 000 nodes, after that it throws a segmentation fault.

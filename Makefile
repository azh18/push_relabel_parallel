all:
	mpic++ -g -std=c++11 -I. main.cpp par_push_relabel_staticAlloc.cpp -o par_push_relabel_staticAlloc 
	mpic++ -g -std=c++11 -I. main.cpp par_push_relabel_dynamicAlloc.cpp -o par_push_relabel_dynamicAlloc 
	mpic++ -g -std=c++11 -I. main.cpp par_push_relabel_2.cpp -o par_push_relabel_2 

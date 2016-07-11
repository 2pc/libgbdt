# libgbdt

Non original ! The code  Forked From Other Place,  and add some Annotation

A implement of LS_Boost in [Greedy Function Approximation: A Gradient Boosting Machine.](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)


```
make clean 
make all
cd output/test
./gbdt-train -r 0.8 -t 100 -s 0.03 -n 30 -d 5 -m test.model -f ../../train
./gbdt-test ./test.model ../../train
```

```
 	{"help", 0, NULL, 'h'},
 	{"sample_feature_ratio", 1, NULL, 'r'},
 	{"tree_num", 1, NULL, 't'},
 	{"shrink", 1, NULL, 's'},
 	{"min_node_size", 1, NULL, 'n'},
 	{"max_depth", 1, NULL, 'd'},
 	{"model_out", 1, NULL, 'm'},
 	{"train_file", 1, NULL, 'f'},
 	{NULL, 0, NULL, 0}};
```

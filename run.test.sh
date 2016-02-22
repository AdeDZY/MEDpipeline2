# Sift RBF
g=(0.1 0.005 0.00001)
for i in {1..3};do
./scripts/train_svm.py P00${i} siftbow_features/all_avg.vectors 500 sift_pred/P00${i}_avg.model -g ${g[i-1]} -k rbf 
./scripts/test_svm.py sift_pred/P00${i}_avg.model siftbow_features/all_avg.vectors 500 sift_pred/P00${i}_avg.pred
../hw1/mAP/ap list/P00${i}_test_label sift_pred/P00${i}_avg.pred >> log
done

# CNN RBF
g=(0.0002 0.0001 0.001)
for i in {1..3};do
./scripts/train_svm.py P00${i} cnn_fc7_features/all_avg.vectors 4096 cnn_pred/fc7_P00${i}.model -k rbf -g ${g[i-1]} 
./scripts/test_svm.py cnn_pred/fc7_P00${i}.model cnn_fc7_features/all_avg.vectors 4096 cnn_pred/fc7_P00${i}.pred
../hw1/mAP/ap list/P00${i}_test_label cnn_pred/fc7_P00${i}.pred >> log
done


# Imtraj linear 
for i in {1..3}; do
./scripts/train_svm.py P00${i} imtraj 32748 imtraj_pred/P00${i}.model -f imtraj 
./scripts/test_svm.py imtraj_pred/P00${i}.model imtraj 32748 imtraj_pred/P00${i}.pred -f imtraj
../hw1/mAP/ap list/P00${i}_test_label imtraj_pred/P00${i}.pred >> log
done

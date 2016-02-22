#for i in {1..3};do
#./scripts/train_svm.py P00${i} siftbow_features/all_max.vectors 500 sift_pred/P00${i}_max.model -g 0.01 -k rbf 
#./scripts/test_svm.py sift_pred/P00${i}_max.model siftbow_features/all_max.vectors 500 sift_pred/P00${i}_max.pred
#../hw1/mAP/ap list/P00${i}_test_label sift_pred/P00${i}_max.pred
#done

#for i in {1..3};do
#./scripts/test_svm.py sift_pred/P00${i}_k500.model siftbow_features/all_max.vectors 500 sift_pred/P00${i}.pred
#../hw1/mAP/ap list/P00${i}_test_label sift_pred/P00${i}_k500.pred
#done

#./scripts/train_svm.py P001 cnn_features/all_avg.vectors 1000 cnn_pred/P001.model -k rbf -g 0.0002
#./scripts/test_svm.py cnn_pred/P001.model cnn_features/all_avg.vectors 1000 cnn_pred/P001.pred
#../hw1/mAP/ap list/P001_test_label cnn_pred/P001.pred
#
#./scripts/train_svm.py P002 cnn_features/all_avg.vectors 1000 cnn_pred/P002.model -k rbf -g 0.0001
#./scripts/test_svm.py cnn_pred/P002.model cnn_features/all_avg.vectors 1000 cnn_pred/P002.pred
#../hw1/mAP/ap list/P002_test_label cnn_pred/P002.pred
#
#./scripts/train_svm.py P003 cnn_features/all_avg.vectors 1000 cnn_pred/P003.model -k rbf -g 0.001
#./scripts/test_svm.py cnn_pred/P003.model cnn_features/all_avg.vectors 1000 cnn_pred/P003.pred
#../hw1/mAP/ap list/P003_test_label cnn_pred/P003.pred

g=(0.0002 0.0001 0.001)
i=0
for i in {1..3};do
./scripts/train_svm.py P00${i} cnn_fc7_features/all_avg.vectors 4096 cnn_pred/fc7_P00${i}.model -k rbf -g ${g[i-1]} 
./scripts/test_svm.py cnn_pred/fc7_P00${i}.model cnn_fc7_features/all_avg.vectors 4096 cnn_pred/fc7_P00${i}.pred
../hw1/mAP/ap list/P00${i}_test_label cnn_pred/fc7_P00${i}.pred
done


#for i in {1..3};do
#./scripts/test_svm.py cnn_pred/P00${i}.model cnn_features/all.vectors 100 cnn_pred/P00${i}.pred
#../hw1/mAP/ap list/P00${i}_test_label cnn_pred/P00${i}.pred
#done

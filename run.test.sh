./scripts/train_svm.py P001 cnn_features/all_avg.vectors 1000 cnn_pred/P001.model -k rbf -g 0.0002
./scripts/test_svm.py cnn_pred/P001.model cnn_features/all_avg.vectors 1000 cnn_pred/P001.pred
../hw1/mAP/ap list/P001_test_label cnn_pred/P001.pred

./scripts/train_svm.py P002 cnn_features/all_avg.vectors 1000 cnn_pred/P002.model -k rbf -g 0.0001
./scripts/test_svm.py cnn_pred/P002.model cnn_features/all_avg.vectors 1000 cnn_pred/P002.pred
../hw1/mAP/ap list/P002_test_label cnn_pred/P002.pred

./scripts/train_svm.py P003 cnn_features/all_avg.vectors 1000 cnn_pred/P003.model -k rbf -g 0.001
./scripts/test_svm.py cnn_pred/P003.model cnn_features/all_avg.vectors 1000 cnn_pred/P003.pred
../hw1/mAP/ap list/P003_test_label cnn_pred/P003.pred


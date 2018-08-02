import FileUtil
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class GBDT_LR:
    def __init__(self):
        '''
         读取one hot encoding之后的数据
        '''
        self.train_x, self.train_y = FileUtil.readCSV('train', './Dataset/train.csv')
        self.test_x = FileUtil.readCSV('test', './Dataset/test.csv')

    def predict(self):
        gbm = lgb.LGBMRegressor(objective='binary',
                                subsample=0.8,
                                min_child_weight=0.5,
                                colsample_bytree=0.7,
                                num_leaves=100,
                                max_depth=12,
                                learning_rate=0.05,
                                n_estimators=10,
                                )

        gbm.fit(self.train_x, self.train_y,
                eval_names=['train', 'val'],
                eval_metric='binary_logloss',
                # early_stopping_rounds = 100,
                )
        model = gbm.booster_
        print('训练得到叶子数')

        # gbdt 预测得到新的特征
        gbdt_feats_train = model.predict(self.train_x, pred_leaf=True)
        gbdt_feats_test = model.predict(self.test_x, pred_leaf=True)

        ohe = OneHotEncoder()

        gbdt_feats_name = ['gbtd_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
        df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
        df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

        index = 0
        # gbdt 特征 one-hot encoding
        for col in gbdt_feats_name:
            ohe_train_result = ohe.fit_transform(df_train_gbdt_feats[col].values.reshape(-1, 1)).toarray()
            ohe_test_result = ohe.fit_transform(df_test_gbdt_feats[col].values.reshape(-1, 1)).toarray()

            if index == 0:
                train_r = ohe_train_result
                test_r = ohe_test_result
                index = index + 1
            else:
                train_r = np.hstack((train_r, ohe_train_result))
                test_r = np.hstack((test_r, ohe_test_result))

        self.train_x = np.hstack((self.train_x, train_r))
        self.test_x = np.hstack((self.test_x, test_r))

        lr = LogisticRegression()
        lr.fit(self.train_x,self.train_y)
       # train_log_loss=log_loss(self.train_y,lr.predict_log_proba(self.train_x))

        predictions=lr.predict(self.test_x)
        df = pd.read_csv("./Dataset/test.csv")
        ids = df["PassengerId"]
        output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
        output.to_csv('titanic-predictions.csv', index=False)





if __name__ == '__main__':
    model = GBDT_LR()
    model.predict()

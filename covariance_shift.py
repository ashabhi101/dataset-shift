import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

class Covariance_shift:
    def __init__(self, train, test, randomstate=142):
        self.train = train
        self.test = test
        self.randomstate = randomstate
        self.train.loc[:,'target']= 1
        self.test.loc[:,'target'] =0
        
        self.X_tmp = pd.concat((self.train, self.test),
                          ignore_index=True).drop(['target'], axis=1)
        self.y_tmp= pd.concat((self.train.target,self.test.target),
                       ignore_index=True)
         # categorical variables need to be label encoded, if it's not alredy done

    def encode_data(self):
        le = LabelEncoder()
        for col in self.X_tmp.columns:
            if self.X_tmp.dtypes[str(col)].name == 'object' or self.X_tmp.dtypes[str(col)].name == 'category':
                self.X_tmp.loc[:,col] = le.fit_transform(self.X_tmp.loc[:,col])
                self.X_tmp.loc[:,col] = self.X_tmp.loc[:,col].astype('category')

    def summary(self):
        Covariance_shift.encode_data(self)
        # Loop over each column in X and calculate ROC-AUC
        drifts = []

        for col in self.X_tmp.columns:
            
            self.X_train_tmp, self.X_test_tmp, \
            self.y_train_tmp, self.y_test_tmp = train_test_split(self.X_tmp[[col]],
                                                   self.y_tmp,
                                                   test_size=0.25,
                                                   random_state=1)
            
            # Use Random Forest classifier
            rf = RandomForestClassifier(n_estimators=50,
                                        n_jobs=-1,
                                        max_features=1.,
                                        min_samples_leaf=5,
                                        max_depth=5,
                                        random_state=1)

            # Fit 
            rf.fit(self.X_train_tmp, self.y_train_tmp)

            # Predict
            self.y_pred_tmp = rf.predict_proba(self.X_test_tmp)[:, 1]

            # Calculate ROC-AUC
            score = roc_auc_score(self.y_test_tmp, self.y_pred_tmp)

            drifts.append((col, (max(np.mean(score), 1 - np.mean(score)) - 0.5) * 2))

        for i in drifts:
            print(i)
#if __name__ == "__main__":

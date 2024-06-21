# Preet Jain
# preetmanutd@gmail.com
# Project


import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils import resample


def preprocess(self, X):

    X['title'] = X['title'].str.strip()
    X['description'] = X['description'].str.strip()
    X['requirements'] = X['description'].str.strip()

    return X


class my_model():
    def fit(self, X, y):
        X = preprocess(self, X)
        X['fraudulent'] = y
        majority_class = X[X['fraudulent'] == 0]
        minority_class = X[X['fraudulent'] == 1]
        minority_class_upsampled = resample(minority_class,
                                            replace=True,  # Sample with replacement
                                            n_samples=len(majority_class),  # Match the number of majority class samples
                                            random_state=9)

        df_train_upsampled = pd.concat([majority_class, minority_class_upsampled])

        X = df_train_upsampled.drop(columns=['fraudulent'])
        y = df_train_upsampled['fraudulent']

        XText = X['title'] + ' ' + X['description'] + ' ' + X['requirements']
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=6000, norm='l2', max_df=0.7)
        textFeat = self.tfidf_vectorizer.fit_transform(XText).toarray()
        X = np.concatenate([textFeat, X[['has_company_logo']].values], axis=1)

        # self.model = RandomForestClassifier(random_state=42)
        self.model = SGDClassifier(class_weight="balanced", max_iter=5000, random_state=9)
        self.model.fit(X, y)

        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions

        X = preprocess(self, X)
        # X_text = X['title'] + ' ' + X['location'] + ' ' + X['description'] + ' ' + X['requirements']
        X_text = X['title'] + ' ' + X['description'] + ' ' + X['requirements']
        text_features = self.tfidf_vectorizer.transform(X_text).toarray()
        X = np.concatenate([text_features, X[['has_company_logo']].values], axis=1)

        predictions = self.model.predict(X)

        return predictions

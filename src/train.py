from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import pandas as pd

def getData(filename, vectorizer, train=True):
    df = pd.read_csv(filename, header=None)
    docs = df.loc[:,0].values.tolist()

    if train:
        X = vectorizer.fit_transform(docs)
    else:
        X = vectorizer.transform(docs)

    y = df.loc[:,1].values

    return X, y, vectorizer

def train(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)

    print("train f1:{0:.4f} test f1:{1:.4f}".format(f1_train, f1_test))


def main():
    vectorizer = TfidfVectorizer(min_df=2)
    X_train, y_train, vectorizer = getData("./train.csv", vectorizer, True)
    X_test, y_test, vectorizer = getData("./test.csv", vectorizer, False)

    train(X_train, y_train, X_test, y_test, MultinomialNB())
    train(X_train, y_train, X_test, y_test, SGDClassifier(random_state=0, n_jobs=-1 ,class_weight="balanced", max_iter=1000))
   
if __name__ == "__main__":
    main()
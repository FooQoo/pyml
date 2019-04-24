from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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

def main():
    vectorizer = TfidfVectorizer()
    X_train, y_train, vectorizer = getData("./train.csv", vectorizer, True)
    X_test, y_test, vectorizer = getData("./test.csv", vectorizer, False)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)

    print("train f1:{0:.4f} test f1:{1:.4f}".format(f1_train, f1_test))


if __name__ == "__main__":
    main()
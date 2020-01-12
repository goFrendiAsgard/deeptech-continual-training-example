import sys
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main(name):
    df = pd.read_csv("./data/" + name + ".csv")
    print(df.head())
    x = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    # train classifier
    clf = GaussianNB()
    clf.fit(x, y)

    # measure accuracy
    accuracy = accuracy_score(y, clf.predict(x))
    print("ACCURACY", accuracy)

    # save accuracy
    with open("./accuracy/log.csv", "a") as accuracy_log:
        accuracy_log.write(name + ", " + str(accuracy))

    # save model
    pickle.dump(clf, open("./model/" + name + ".model", "wb"))
    pickle.dump(clf, open("./model/latest.model", "wb"))

if  __name__ == "__main__":
    main(sys.argv[1])

from sklearn.svm import SVC

def create_svm():

    model = SVC(
        kernel="rbf",
        C=5,
        gamma="scale"
    )

    return model

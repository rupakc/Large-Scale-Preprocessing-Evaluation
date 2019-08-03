from sklearn.model_selection import train_test_split


def get_train_test_split(features, labels_or_values, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(features, labels_or_values, test_size=test_size)
    return X_train, X_test, y_train, y_test

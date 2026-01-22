from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## Chia tập train và test:
def split(str1, str2, test_size, random_state):
    label_encode = LabelEncoder()
    y_data = label_encode.fit_transform(str1)
    X_train, X_test, y_train, y_test = train_test_split(str2, y_data, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
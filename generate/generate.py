import os
import numpy as np
from skimage import io, transform
from skimage.color import rgba2rgb, gray2rgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys

####### CONST                                        #
PATH = 'color_clasifff.pkl'
IMG_SIZE = (32, 32)

def clear_screen():          # +
    if os.name == 'nt':
        os.system('cls')

def prepare_dataset(root_dir="generate/dataset", save_npy=False):      # +
    X_list, Y_list = [], []

    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(label_dir, fname)
            img = io.imread(path)

            if img.shape[-1] == 4:
                img = rgba2rgb(img)
            if len(img.shape) == 2:
                img = gray2rgb(img)

            img_resized = transform.resize(img, IMG_SIZE, anti_aliasing=True)
            img_array = (img_resized * 255).astype(np.uint8)

            X_list.append(img_array.flatten())
            Y_list.append(label)

    X = np.array(X_list)
    Y = np.array(Y_list)
    encoder = LabelEncoder()


    if save_npy:    
        np.save("X.npy", X)
        np.save("Y.npy", Y)
        np.save("classes.npy", encoder.classes_)

    return X, Y

def train_save():                      # +-
    X, Y = prepare_dataset()


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



    #$            Zapis logov           $#

    # np.set_printoptions(threshold=sys.maxsize)

    # with open('log.txt', 'w') as file:
    #     file.write(np.array2string(X, max_line_width=sys.maxsize))
    # with open('log.txt', 'w') as file:
    #     file.write(np.array2string(Y))
    
    # print(Y)

    le = LabelEncoder()
    Y = le.fit_transform(Y)

    # print(Y)
    # print(dict(zip(le.classes_, le.transform(le.classes_)))) # type: ignore
    
    # model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)


    #$ TEST        F1  AC                    $#

    # y_pred = model.predict(X_train)
    # print('Accuracy:', accuracy_score(y_pred, y_train))
    # print(y_pred)
    # print(y_train)

    return model

def photo2np(path):

    X = []

    img = io.imread(path)
    if img.shape[-1] == 4:
        img = rgba2rgb(img)
    if len(img.shape) == 2:
        img = gray2rgb(img)

    img_resized = transform.resize(img, IMG_SIZE, anti_aliasing=True)
    img_array = (img_resized * 255).astype(np.uint8)

    X.append(img_array.flatten())
    return X

def predict_color(model, path):

    arr = photo2np(path)
    prediction = model.predict(arr)
    return prediction

def model_save():
    model = train_save()
    joblib.dump(model, PATH)


if __name__ == '__main__':
    clear_screen()
    model_save()
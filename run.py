
import numpy as np
img_size_224p = 128

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array




def load_image(filename):
    img = load_img(filename, target_size=(img_size_224p, img_size_224p))
    # plt.imshow(img)
    # plt.axis("off");
    img = img_to_array(img)
    img = img.reshape(-1, img_size_224p, img_size_224p, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img

def maxnumber(model_classification,path_img):
    img_resize = load_image(path_img)
    pred = model_classification.predict(img_resize)
    print(pred)
    for numlist in pred:
        maxnum = max(numlist) * 100
    return round(maxnum, 2)

def run_predict(model_classification,path_img):
    img_resize = load_image(path_img)
    pred = model_classification.predict(img_resize)
    print(pred)
    for numlist in pred:
        maxnum = max(numlist)*100
        print(round(maxnum,2))
    prediksi = np.argmax(pred,axis=-1)  # Try -> predict.shape -> (800, 5) -> axis = -1 it will get that value 5 (number of Orchid Labels)

    label = ""

    if(prediksi[0] == 0):
        print("\nPredict Label: Cattleya")
        label = "Cattleya"
    elif prediksi[0] == 1:
        print("\nPredict Label: Dendrobium")
        label = "Dendrobium"
    elif prediksi[0] == 2:
        print("\nPredict Label: Oncidium")
        label = "Oncidium"
    elif prediksi[0] == 3:
        print("\nPredict Label: Phalaenopsis")
        label = "Phalaenopsis"
    elif prediksi[0] == 4:
        print("\nPredict Label: Vanda")
        label = "Vanda"

    return label
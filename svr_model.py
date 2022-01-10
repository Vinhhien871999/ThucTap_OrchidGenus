
from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from run import run_predict
from run import maxnumber
model_classification = load_model(r'C:\Users\Nguyen Vinh Hien\PycharmProjects\TrainOrchild\model_without_kfold.h5') # load đường dẫn file weight sau khi train

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"


# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == "GET":
        return render_template('index.html')
    else:
        image_file = request.files['file']
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(path_to_save)
        print(path_to_save)
        predict_label = run_predict(model_classification,path_to_save)
        maxthenumber = maxnumber(model_classification,path_to_save)
        return render_template("index.html", user_image= image_file.filename, msg= "Tải Thành Công",
                                   face_shape= predict_label,
                                   hasface= True,
                                   extra = maxthenumber,
                                   button1 = "Có",
                                   button2="không"
                               )
    # else:
    #     return render_template('index.html', msg='Không nhận diện được cây')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000, debug=True)
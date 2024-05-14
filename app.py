from flask import Flask,jsonify,request
from RandomForest import RandomForest
from DecisionTree import DecisionTree
from node import Node
import pickle

app = Flask(__name__)

def loadModel():
    # Memuat model RandomForest dari file pickle
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def prediksi(data,Model):
    HasilPrediksi=Model.predict(data)
    if HasilPrediksi[0] == 0:
        return 'Rendah'
    elif HasilPrediksi[0] ==1:
        return 'Sedang'
    else :
        return 'Tinggi'


@app.route('/')
def hello_world():
    return 'Api siap digunakan'

@app.route('/prediksiRandomForest', methods=['POST'])
def prediksii():
    if request.method == 'POST':
         a= request.form.get('a')
         b= request.form.get('b')
         c= request.form.get('c')
         print(a)
         #  ubah ke float
         a= float(a)
         b= float(b)
         c= float(c)
         data=[[a,b,c]]
        #  Load Model
         Model=loadModel()
         HasilPrediksi=prediksi(data,Model)

         return jsonify({
            'Klasifikasi':HasilPrediksi
            }),200

if __name__ =='__main__':
	#app.debug = True
	app.run(debug=True)






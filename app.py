#import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_material import Material
import pickle
import pandas as pd
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)
Material(app)


@app.route('/')
def index():
	return render_template("index.html")	

@app.route('/preview')
def preview():
	df = pd.read_csv("data/50_Startups.csv")
	return render_template("preview.html",df_view = df)

@app.route('/analyze',methods=["POST"])
def analyze():
	if request.method == 'POST':
		Research_Spend = request.form['Research_Spend']
		Administration = request.form['Administration']
		Marketing_Spend = request.form['Marketing_Spend']
		State = request.form['State']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [Research_Spend,Administration,Marketing_Spend,State]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)


		if model_choice == 'linearmodel':
		    linear_model = joblib.load('data/model.pkl')
		    result_prediction = linear_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'svmmodel':
			knn_model = joblib.load('data/svm_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)


		return render_template('index.html', Research_Spend=Research_Spend,
		Administration=Administration,
		Marketing_Spend=Marketing_Spend,
		State=State,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)

if __name__ == "__main__":
    app.run(debug=True)

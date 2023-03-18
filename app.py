from flask import Flask,request,render_template
import numpy  as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

# Route for home page
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediction',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
        gender = request.form.get('gender'),
        ethnicity = request.form.get('ethnicity'),
        lunch = request.form.get('lunch'),
        parental_level_of_education = request.form.get('parental_level_of_education'),
        test_preparation_course = request.form.get('test_preparation_course'),
        reading_score = float(request.form.get('reading_score')),
        writing_score = float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pilpeline = PredictPipeline()
        results = predict_pilpeline.predict(pred_df)
        return render_template('home.html',results = results[0])


if __name__=="main":
    app.run(host="0.0.0.0",debug=True)
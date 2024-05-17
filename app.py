from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('index.html')
    else:
        data = CustomData(
        gender=request.form.get('gender'),
        tenure_months=int(request.form.get('tenure_months')),
        monthly_charges=float(request.form.get('monthly_charges')),
        total_charges=float(request.form.get('total_charges')),
        senior_citizen=request.form.get('senior_citizen'),
        partner=request.form.get('partner'),
        dependents=request.form.get('dependents'),
        phone_service=request.form.get('phone_service'),
        multiple_lines=request.form.get('multiple_lines'),
        internet_service=request.form.get('internet_service'),
        online_security=request.form.get('online_security'),
        online_backup=request.form.get('online_backup'),
        device_protection=request.form.get('device_protection'),
        tech_support=request.form.get('tech_support'),
        streaming_tv=request.form.get('streaming_tv'),
        streaming_movies=request.form.get('streaming_movies'),
        contract=request.form.get('contract'),
        paperless_billing=request.form.get('paperless_billing'),
        payment_method=request.form.get('payment_method')
    )
    
    pred_df = data.get_data_as_frame()
    print(pred_df)
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('index.html', results=results[0])
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
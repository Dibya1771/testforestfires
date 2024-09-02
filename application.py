import pickle
from flask import Flask, request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

## Import ridge regressor and standarScaler model (That is loade the pickle file)
ridge_model=pickle.load(open('models/ridge.pkl','rb')) # Read Byte mode
standard_scaler=pickle.load(open('models/scaler.pkl','rb')) # Read Byte mode


application = Flask(__name__)
app=application

# Home page
@app.route("/")
def index():
    return render_template('index.html')


# For prediction
@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        ##1-  Getting all parameter
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        ## 2- Applying standardization to Input features (give in 2D format)
        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        ## 3- Predict result
        result = ridge_model.predict(new_data_scaled)

        ## Showing this result in frontend (so assign to a variable and call in home.html page)
        return render_template('home.html',results=result[0])

    else:
        return render_template("home.html")
        

if __name__=="__main__":
    app.run(host="0.0.0.0")

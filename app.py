import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('C:\\Users\\PC\\Desktop\\fwdpricepredictionmodel\\model.pkl', 'rb'))
sc = pickle.load(open('C:\\Users\\PC\\Desktop\\fwdpricepredictionmodel\\standardized1.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data={}
 
    data['fpl_value'] = [str(request.form.get('fpl_value'))]
    data['fpl_sel'] = [int(request.form.get('fpl_sel'))]
    data['fpl_points'] = [str(request.form.get('fpl_points'))]
    #data['new_foreign'] = [str(request.form.get('new_foreign'))]
    data['age_cat'] = [str(request.form.get('age_cat'))]
    data['big_club'] = [str(request.form.get('big_club'))]
    
    #data['fpl_sel']= data['fpl_sel']/100
    
    data=pd.DataFrame.from_dict(data)
    data['fpl_sel']= data['fpl_sel']/100
    #print(data)
    
    scaled = sc.transform(data)
    prediction = model.predict(scaled)
    
#    int_features = [int(x) for x in request.form.values()]
#    scaled_features = sc.transform(int_features)
#    final_features = [np.array(scaled_features)].to_numpy()
    #prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Market price of player should be $ {}'.format(output))

#@app.route('/predict_api',methods=['POST'])
#def predict_api():
#    '''
#    For direct API calls trought request
#    '''
#    data = request.get_json(force=True)
#    prediction = model.predict([np.array(list(data.values()))])
#
#    output = prediction[0]
#    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    
 

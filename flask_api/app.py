from flask import Flask, request, jsonify
import pickle 
import numpy as np
import pandas as pd

app = Flask(__name__)

loaded_model = pickle.load(open("consumption_model.pkl", "rb"))
loaded_full_pipeline = pickle.load(open("full_pipeline.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    
    columns = ['hydro_share_energy', 'other_renewables_share_energy', 'country', 'greenhouse_gas_emissions', 'nuclear_consumption', 'energy_per_capita', 'solar_consumption', 'energy_consumption', 'other_renewable_consumption', 'per_capita_electricity', 'solar_share_energy', 'hydro_consumption', 'nuclear_share_energy', 'wind_share_energy', 'fossil_energy_consumption', 'electricity_generation', 'year', 'biofuel_share_energy', 'fossil_share_elec', 'wind_consumption', 'biofuel_consumption', 'fossil_share_energy', 'gdp_millions', 'population', 'primary_energy_consumption', 'renewable_share_energy']
    
    df = pd.DataFrame([data['features']], columns=columns)
    
    X_test_prepared = loaded_full_pipeline.transform(df)
    
    prediction = loaded_model.predict(X_test_prepared)
    
    return jsonify({'prediction':prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
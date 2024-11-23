import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features":[0.30, 0.05, 'Brasil', 1200, 0.03, 2700, 0.02, 90000, 3000, 2200, 0.04, 30000, 0.01, 0.06, 55000, 85000, 2023, 0.10, 0.65, 2500, 5000, 0.55, 2000000, 213000000, 120000, 0.45]
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Previsão", response.json()['predictions'])
else:
    print("Erro:", response.json())
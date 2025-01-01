import threading

import jsons
from flask import Flask, jsonify, request
from pyngrok import ngrok

from cashflow_model import process_cashflow_data

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "Invalid input, JSON expected"}), 400
        
        print(json_data)
        results = process_cashflow_data(jsons.dumps(json_data))
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    app.run(port=5000)

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"ngrok tunnel: {public_url}")
    threading.Thread(target=run_flask).start()

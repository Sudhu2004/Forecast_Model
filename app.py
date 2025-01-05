import threading

import json
from flask import Flask, jsonify, request
from pyngrok import ngrok

from cashflow_model import process_cashflow_data
import time

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # try:
    #     json_data = request.data.decode("utf-8")
    #     result = process_cashflow_data(json_data)
    #     return jsonify({"status": "success", "data": result}), 200
    # except Exception as e:
    #     print(f"Current Exception : {str(e)}")
    #     return jsonify({"status": "error", "message": str(e)}), 400

    start = time.time()
    data = request.json  # Receive the JSON data
    if isinstance(data, list):
        try:
            json_data = json.dumps([
                {
                    "date": entry.get("date"),
                    "revenueSales": entry.get("revenueSales"),
                    "receivables": entry.get("receivables"),
                    "expenses": entry.get("expenses"),
                    "debts": entry.get("debts"),
                    "netCashFlow": entry.get("netCashFlow")
                }
                for entry in data
            ])
            # print(f"Json Data: {json_data}")
            print("Initializing CashFlow data")
            result = process_cashflow_data(json_data)
            print("Success Data processed")
            print(f"Time Taken: {time.time() - start}")
            return jsonify({"status": "success", "data": result}), 200
        except Exception as e:
            print(f"Current Exception : {str(e)}")
            print(f"Time Taken: {time.time() - start}")
            return jsonify({"status": "error", "data": str(e)}), 400

    else:
        print(f"Time Taken: {time.time() - start}")
        return jsonify({"status": "error", "data": {}}), 400
    
    
    
def run_flask():
    app.run(port=5000)

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"ngrok tunnel: {public_url}")
    threading.Thread(target=run_flask).start()

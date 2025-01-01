from itertools import product
from typing import List

import jsons
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class CashFlowData:
    def __init__(self, id, date, revenueSales, receivables, expenses, debts, netCashFlow):
        self.id = id
        self.date = date
        self.revenueSales = revenueSales
        self.receivables = receivables
        self.expenses = expenses
        self.debts = debts
        self.netCashFlow = netCashFlow

def process_cashflow_data(json_data):
    cashflow_data = jsons.loads(json_data, List[CashFlowData])
    time_series = [entry.netCashFlow for entry in cashflow_data]
    train_size = len(time_series) - 15
    train_data = time_series[:train_size]
    test_data = time_series[train_size:]

    p_range, d_range, q_range = range(0, 3), [0, 1], range(0, 3)
    P_range, D_range, Q_range = range(0, 2), [0, 1], range(0, 2)
    seasonal_period = 7
    best_rmse = float('inf')
    best_params = None
    best_forecast = None

    def fit_sarima_and_evaluate(train, test, order, seasonal_order):
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=len(test))
            predicted_mean = forecast.predicted_mean
            rmse = np.sqrt(np.mean((np.array(test) - np.array(predicted_mean)) ** 2))
            return rmse, results
        except Exception as e:
            return float('inf'), None

    for p, d, q, P, D, Q in product(p_range, d_range, q_range, P_range, D_range, Q_range):
        try:
            order = (p, d, q)
            seasonal_order = (P, D, Q)
            rmse, model = fit_sarima_and_evaluate(train_data, test_data, order, seasonal_order)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (order, seasonal_order)
                best_forecast = model
        except Exception as e:
            continue

    if best_forecast:
        forecast = best_forecast.get_forecast(steps=30)
        forecast_values = forecast.predicted_mean.tolist()
        return {
            "predictions": forecast_values,
            "best_order": {
                "order": best_params[0],
                "seasonal_order": best_params[1]
            },
            "rmse": best_rmse,
        }
    else:
        raise RuntimeError("Could not fit any SARIMA model. Check the input data or parameter ranges.")

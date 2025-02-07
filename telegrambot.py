import os
from dotenv import load_dotenv
load_dotenv()

import tensorflow as tf
import tensorflow_probability as tfp
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WILDBERRIES_API_KEY = os.getenv("WILDBERRIES_API_KEY")

# TensorFlow Bayesian Neural Network
def bayesian_dense(units, activation):
    return tfp.layers.DenseVariational(
        units,
        make_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        make_prior_fn=tfp.layers.default_mean_field_normal_fn(),
        activation=activation
    )

def create_bnn(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = bayesian_dense(64, activation="relu")(inputs)
    x = bayesian_dense(32, activation="relu")(x)
    outputs = bayesian_dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

model = create_bnn(input_dim=10)

def custom_loss(y_true, y_pred, kl_divergence, weight=1e-3):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    kl_penalty = weight * kl_divergence
    return mse + kl_penalty

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, model.losses[0])
)

# Wildberries API data fetching
def fetch_wb_data(api_key, date_from, date_to):
    url = "https://suppliers-api.wildberries.ru/api/v1/sales"
    headers = {"Authorization": api_key}
    params = {"dateFrom": date_from, "dateTo": date_to, "limit": 1000}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")

# Data cleaning and transformation
def remove_outliers(dataframe, column):
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_columns(dataframe, columns):
    scaler = MinMaxScaler()
    dataframe[columns] = scaler.fit_transform(dataframe[columns])
    return dataframe

def aggregate_time_series(dataframe, date_column, metric_column, freq="D"):
    dataframe[date_column] = pd.to_datetime(dataframe[date_column])
    return dataframe.groupby(pd.Grouper(key=date_column, freq=freq))[metric_column].sum().reset_index()

# PyTorch + Pyro Bayesian block
class BayesianRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight_mean = nn.Parameter(torch.zeros(input_dim))
        self.weight_scale = nn.Parameter(torch.ones(input_dim))
        self.bias_mean = nn.Parameter(torch.zeros(1))
        self.bias_scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        weight = pyro.sample("weight", dist.Normal(self.weight_mean, self.weight_scale))
        bias = pyro.sample("bias", dist.Normal(self.bias_mean, self.bias_scale))
        return torch.matmul(x, weight) + bias

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.bayesian_block = BayesianRegression(input_dim)
        self.deep_block = DeepNeuralNetwork(input_dim, hidden_dim)
    def forward(self, x):
        bayesian_output = self.bayesian_block(x)
        final_output = self.deep_block(bayesian_output)
        return final_output

def generate_recommendations(predictions, thresholds):
    recommendations = []
    for metric, value in predictions.items():
        if value < thresholds[metric]:
            recommendations.append(f"Increase the budget for {metric}, current: {value:.2f}")
        else:
            recommendations.append(f"{metric} is within acceptable range: {value:.2f}")
    return recommendations

# FastAPI app
app = FastAPI()

class MetricsRequest(BaseModel):
    costs: float
    revenue: float
    returns: float

@app.post("/calculate_metrics/")
async def calculate_metrics(request: MetricsRequest):
    try:
        cac = request.costs / max(request.revenue, 1)
        roi = (request.revenue - request.costs) / max(request.costs, 1)
        ltv = request.revenue * 1.2
        return {
            "CAC": round(cac, 2),
            "ROI": round(roi, 2),
            "LTV": round(ltv, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")

# Telegram bot part
API_URL = "http://127.0.0.1:8000/calculate_metrics/"

def calculate(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    bot: Bot = context.bot
    try:
        costs, revenue, returns = map(float, context.args)
        data = {"costs": costs, "revenue": revenue, "returns": returns}
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            metrics = response.json()
            message = (
                f"Metrics:\n"
                f"✔️ CAC: {metrics['CAC']}\n"
                f"✔️ ROI: {metrics['ROI']}\n"
                f"✔️ LTV: {metrics['LTV']}"
            )
        else:
            message = f"Error: {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        message = f"Invalid data or error: {str(e)}"
    bot.send_message(chat_id=chat_id, text=message)

def main():
    updater = Updater(TELEGRAM_TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("calculate", calculate))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
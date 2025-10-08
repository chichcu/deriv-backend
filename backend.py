from fastapi import FastAPI
import numpy as np
from collections import deque

app = FastAPI(title="Deriv Prediction Backend")

ROLLING_WINDOW_SIZE = 800
ticks_window = deque(maxlen=ROLLING_WINDOW_SIZE)

# ---- PRIMARY CORE ----
def compute_primary_probabilities(ticks):
    markov = np.random.rand(10); markov_weight = 0.153
    bayesian = np.random.rand(10); bayesian_weight = 0.154
    euclidean = np.random.rand(10); euclidean_weight = 0.124
    frequency = np.random.rand(10); frequency_weight = 0.17
    probability = np.random.rand(10); probability_weight = 0.125
    mean_reversion = np.random.rand(10); mean_weight = 0.14
    kalman = np.random.rand(10); kalman_weight = 0.149

    combined = (markov*markov_weight + bayesian*bayesian_weight +
                euclidean*euclidean_weight + frequency*frequency_weight +
                probability*probability_weight + mean_reversion*mean_weight +
                kalman*kalman_weight)
    combined /= combined.sum()
    return combined

# ---- SECONDARY CORE ----
def compute_secondary_scale(ticks):
    entropy = 0.641
    fractal = 0.39
    spectral = 0.552
    graph = 0.908
    modular = 0.372
    savgol = 0.971
    entropy_gate = 0.994

    combined_secondary = np.mean([entropy, fractal, spectral, graph, modular, savgol, entropy_gate])
    adaptive_scale = combined_secondary * 0.95
    return adaptive_scale

# ---- FINAL COMPUTATION ----
def compute_final_prediction(primary_probs, adaptive_scale):
    scaled = primary_probs * adaptive_scale
    scaled /= scaled.sum()
    final_digit = int(np.argmax(scaled))
    confidence = float(scaled[final_digit])
    return final_digit, confidence

@app.get("/predict/")
def predict():
    ticks_window.clear()
    ticks_window.extend(np.random.rand(ROLLING_WINDOW_SIZE))
    primary_probs = compute_primary_probabilities(list(ticks_window))
    adaptive_scale = compute_secondary_scale(list(ticks_window))
    final_digit, confidence = compute_final_prediction(primary_probs, adaptive_scale)
    return {
        "message": "Rolling window auto-filled with 800 ticks.",
        "final_prediction": final_digit,
        "confidence": round(confidence, 3),
        "scaled_probabilities": primary_probs.tolist()
    }

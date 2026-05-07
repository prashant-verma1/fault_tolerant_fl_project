"""
Fault-Tolerant Federated Learning Engine
Refactored from main.py to be importable and return structured results.
"""

import random
import numpy as np
import tensorflow as tf


# -----------------------------
# Model Creation
# -----------------------------
def create_model(learning_rate=0.01):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# -----------------------------
# Data Preparation
# -----------------------------
def load_and_split_data(num_clients):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    data_per_client = len(x_train) // num_clients
    clients_data = []

    for i in range(num_clients):
        start = i * data_per_client
        end = start + data_per_client
        clients_data.append((x_train[start:end], y_train[start:end]))

    return clients_data, (x_test, y_test)


# -----------------------------
# Client Class
# -----------------------------
class Client:
    def __init__(self, client_id, data):
        self.client_id = client_id
        self.x_train, self.y_train = data

        self.participation_count = random.randint(1, 10)
        self.network_quality = random.uniform(0.5, 1.0)
        self.response_time = random.uniform(1.0, 5.0)
        self.reliability_score = self.calculate_reliability()

    def calculate_reliability(self, alpha=0.4, beta=0.4, gamma=0.2):
        participation = self.participation_count / 10
        network = self.network_quality
        response = 1 / self.response_time
        score = alpha * participation + beta * network + gamma * response
        return score

    def local_train(self, global_weights, learning_rate=0.01, local_epochs=1, batch_size=32):
        local_model = create_model(learning_rate)
        local_model.set_weights(global_weights)

        local_model.fit(
            self.x_train,
            self.y_train,
            epochs=local_epochs,
            batch_size=batch_size,
            verbose=0
        )
        return local_model.get_weights()


# -----------------------------
# Server Functions
# -----------------------------
def select_clients(clients, k):
    clients_sorted = sorted(clients, key=lambda c: c.reliability_score, reverse=True)
    return clients_sorted[:k]


def simulate_dropout(selected_clients, dropout_rate):
    active_clients = []
    dropped_clients = []

    for client in selected_clients:
        if random.random() < dropout_rate:
            dropped_clients.append(client)
        else:
            active_clients.append(client)

    return active_clients, dropped_clients


def adaptive_aggregation(client_weights, reliability_scores):
    total_reliability = sum(reliability_scores)

    if total_reliability == 0:
        normalized_scores = [1 / len(reliability_scores)] * len(reliability_scores)
    else:
        normalized_scores = [score / total_reliability for score in reliability_scores]

    new_weights = []

    for weights_tuple in zip(*client_weights):
        weighted_sum = np.zeros_like(weights_tuple[0])
        for client_weight, score in zip(weights_tuple, normalized_scores):
            weighted_sum += client_weight * score
        new_weights.append(weighted_sum)

    return new_weights


# -----------------------------
# Main Training Function
# -----------------------------
def run_training(config: dict) -> dict:
    """
    Run fault-tolerant federated learning with the given config.

    Args:
        config: dict with keys:
            - num_clients (int)
            - selected_clients (int)
            - rounds (int)
            - dropout_rate (float)
            - learning_rate (float)
            - local_epochs (int)
            - batch_size (int)

    Returns:
        dict with:
            - rounds: list of dicts with per-round results
            - clients: list of client info dicts
            - config: the config used
    """
    num_clients = config.get("num_clients", 20)
    selected_k = config.get("selected_clients", 8)
    rounds = config.get("rounds", 10)
    dropout_rate = config.get("dropout_rate", 0.25)
    learning_rate = config.get("learning_rate", 0.01)
    local_epochs = config.get("local_epochs", 1)
    batch_size = config.get("batch_size", 32)

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    clients_data, test_data = load_and_split_data(num_clients)
    x_test, y_test = test_data

    clients = [Client(i, data) for i, data in enumerate(clients_data)]
    global_model = create_model(learning_rate)

    round_results = []

    for round_num in range(1, rounds + 1):
        global_weights = global_model.get_weights()

        selected = select_clients(clients, selected_k)
        active, dropped = simulate_dropout(selected, dropout_rate)

        client_updates = []
        reliability_scores = []

        for client in active:
            updated_weights = client.local_train(
                global_weights, learning_rate, local_epochs, batch_size
            )
            client_updates.append(updated_weights)
            reliability_scores.append(client.reliability_score)

        if len(client_updates) > 0:
            new_global_weights = adaptive_aggregation(client_updates, reliability_scores)
            global_model.set_weights(new_global_weights)

        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)

        round_results.append({
            "round": round_num,
            "accuracy": float(round(accuracy, 4)),
            "loss": float(round(loss, 4)),
            "selected_count": len(selected),
            "active_count": len(active),
            "dropped_count": len(dropped),
            "active_client_ids": [c.client_id for c in active],
            "dropped_client_ids": [c.client_id for c in dropped],
        })

    # Client info summary
    client_info = []
    for c in clients:
        client_info.append({
            "id": c.client_id,
            "reliability_score": round(c.reliability_score, 4),
            "participation_count": c.participation_count,
            "network_quality": round(c.network_quality, 4),
            "response_time": round(c.response_time, 4),
        })

    return {
        "rounds": round_results,
        "clients": client_info,
        "config": config,
    }

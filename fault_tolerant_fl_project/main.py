import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
NUM_CLIENTS = 20
SELECTED_CLIENTS = 8
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
DROPOUT_RATE = 0.25
LEARNING_RATE = 0.01

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# -----------------------------
# Model Creation
# -----------------------------
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
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

    def local_train(self, global_weights):
        local_model = create_model()
        local_model.set_weights(global_weights)

        local_model.fit(
            self.x_train,
            self.y_train,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        return local_model.get_weights()


# -----------------------------
# Server Functions
# -----------------------------
def select_clients(clients, k):
    clients = sorted(clients, key=lambda c: c.reliability_score, reverse=True)
    return clients[:k]


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
# Federated Training
# -----------------------------
def federated_training():
    clients_data, test_data = load_and_split_data(NUM_CLIENTS)
    x_test, y_test = test_data

    clients = [Client(i, data) for i, data in enumerate(clients_data)]
    global_model = create_model()

    accuracy_history = []
    loss_history = []
    dropout_history = []

    print("Starting Fault-Tolerant Federated Learning...\n")

    for round_num in range(1, ROUNDS + 1):
        print(f"Round {round_num}/{ROUNDS}")

        global_weights = global_model.get_weights()

        selected_clients = select_clients(clients, SELECTED_CLIENTS)
        active_clients, dropped_clients = simulate_dropout(selected_clients, DROPOUT_RATE)

        print(f"Selected Clients: {len(selected_clients)}")
        print(f"Active Clients: {len(active_clients)}")
        print(f"Dropped Clients: {len(dropped_clients)}")

        client_updates = []
        reliability_scores = []

        for client in active_clients:
            updated_weights = client.local_train(global_weights)
            client_updates.append(updated_weights)
            reliability_scores.append(client.reliability_score)

        if len(client_updates) > 0:
            new_global_weights = adaptive_aggregation(client_updates, reliability_scores)
            global_model.set_weights(new_global_weights)
        else:
            print("All selected clients dropped out. Skipping aggregation.")

        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        accuracy_history.append(accuracy)
        loss_history.append(loss)
        dropout_history.append(len(dropped_clients))

        print(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}\n")

    save_graphs(accuracy_history, loss_history, dropout_history)
    print("Training completed. Graphs saved in results folder.")


# -----------------------------
# Graphs
# -----------------------------
def save_graphs(accuracy_history, loss_history, dropout_history):
    rounds = range(1, len(accuracy_history) + 1)

    plt.figure()
    plt.plot(rounds, accuracy_history, marker="o")
    plt.xlabel("Training Rounds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Training Rounds")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "accuracy_vs_rounds.png"))
    plt.close()

    plt.figure()
    plt.plot(rounds, loss_history, marker="o")
    plt.xlabel("Training Rounds")
    plt.ylabel("Loss")
    plt.title("Loss vs Training Rounds")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "loss_vs_rounds.png"))
    plt.close()

    plt.figure()
    plt.plot(rounds, dropout_history, marker="o")
    plt.xlabel("Training Rounds")
    plt.ylabel("Number of Dropped Clients")
    plt.title("Client Dropout per Round")
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "dropout_per_round.png"))
    plt.close()


if __name__ == "__main__":
    federated_training()

"""
FastAPI backend for Fault-Tolerant Federated Learning.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fl_engine import run_training

app = FastAPI(
    title="Fault-Tolerant FL API",
    description="API for running fault-tolerant federated learning simulations",
    version="1.0.0",
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainConfig(BaseModel):
    num_clients: int = Field(default=20, ge=2, le=100, description="Total number of clients")
    selected_clients: int = Field(default=8, ge=1, le=100, description="Number of clients selected per round")
    rounds: int = Field(default=10, ge=1, le=50, description="Number of training rounds")
    dropout_rate: float = Field(default=0.25, ge=0.0, le=1.0, description="Client dropout probability")
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0, description="Learning rate for SGD")
    local_epochs: int = Field(default=1, ge=1, le=10, description="Local training epochs per client")
    batch_size: int = Field(default=32, ge=8, le=256, description="Batch size for local training")


@app.get("/api/status")
def health_check():
    return {"status": "ok", "message": "FL Backend is running"}


@app.post("/api/train")
def train(config: TrainConfig):
    """Run federated learning training with the given configuration."""
    result = run_training(config.model_dump())
    return result

"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";

const API_URL = "https://fl-backend-q3v0.onrender.com";

interface RoundResult {
  round: number;
  accuracy: number;
  loss: number;
  selected_count: number;
  active_count: number;
  dropped_count: number;
  active_client_ids: number[];
  dropped_client_ids: number[];
}

interface ClientInfo {
  id: number;
  reliability_score: number;
  participation_count: number;
  network_quality: number;
  response_time: number;
}

interface TrainResult {
  rounds: RoundResult[];
  clients: ClientInfo[];
  config: Record<string, number>;
}

export default function Dashboard() {
  const [config, setConfig] = useState({
    num_clients: 20,
    selected_clients: 8,
    rounds: 10,
    dropout_rate: 0.25,
    learning_rate: 0.01,
    local_epochs: 1,
    batch_size: 32,
  });

  const [result, setResult] = useState<TrainResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const finalAcc = result?.rounds[result.rounds.length - 1]?.accuracy;
  const finalLoss = result?.rounds[result.rounds.length - 1]?.loss;
  const totalDropped = result?.rounds.reduce((s, r) => s + r.dropped_count, 0);

  return (
    <main className="flex-1 p-4 md:p-8 max-w-7xl mx-auto w-full space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight bg-gradient-to-r from-[oklch(0.75_0.18_260)] to-[oklch(0.7_0.2_300)] bg-clip-text text-transparent">
          Federated Learning Dashboard
        </h1>
        <p className="text-muted-foreground text-lg">
          Configure and run fault-tolerant federated learning simulations
        </p>
      </div>

      <Separator />

      {/* Config + Summary Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Card */}
        <Card className="lg:col-span-2 glow-primary">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
              Training Configuration
            </CardTitle>
            <CardDescription>
              Adjust parameters for the FL simulation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Num Clients */}
              <div className="space-y-2">
                <Label htmlFor="num_clients">Total Clients</Label>
                <Input
                  id="num_clients"
                  type="number"
                  min={2}
                  max={100}
                  value={config.num_clients}
                  onChange={(e) =>
                    setConfig({ ...config, num_clients: +e.target.value })
                  }
                />
              </div>
              {/* Selected Clients */}
              <div className="space-y-2">
                <Label htmlFor="selected_clients">Selected per Round</Label>
                <Input
                  id="selected_clients"
                  type="number"
                  min={1}
                  max={config.num_clients}
                  value={config.selected_clients}
                  onChange={(e) =>
                    setConfig({ ...config, selected_clients: +e.target.value })
                  }
                />
              </div>
              {/* Rounds */}
              <div className="space-y-2">
                <Label htmlFor="rounds">Training Rounds</Label>
                <Input
                  id="rounds"
                  type="number"
                  min={1}
                  max={50}
                  value={config.rounds}
                  onChange={(e) =>
                    setConfig({ ...config, rounds: +e.target.value })
                  }
                />
              </div>
              {/* Batch Size */}
              <div className="space-y-2">
                <Label htmlFor="batch_size">Batch Size</Label>
                <Input
                  id="batch_size"
                  type="number"
                  min={8}
                  max={256}
                  step={8}
                  value={config.batch_size}
                  onChange={(e) =>
                    setConfig({ ...config, batch_size: +e.target.value })
                  }
                />
              </div>
              {/* Dropout Rate Slider */}
              <div className="space-y-2 sm:col-span-2">
                <div className="flex justify-between">
                  <Label>Dropout Rate</Label>
                  <Badge variant="secondary">
                    {(config.dropout_rate * 100).toFixed(0)}%
                  </Badge>
                </div>
                <Slider
                  min={0}
                  max={100}
                  step={5}
                  value={[config.dropout_rate * 100]}
                  onValueChange={(val) => {
                    const v = Array.isArray(val) ? val[0] : val;
                    setConfig({ ...config, dropout_rate: v / 100 });
                  }}
                />
              </div>
              {/* Learning Rate Slider */}
              <div className="space-y-2 sm:col-span-2">
                <div className="flex justify-between">
                  <Label>Learning Rate</Label>
                  <Badge variant="secondary">{config.learning_rate}</Badge>
                </div>
                <Slider
                  min={1}
                  max={100}
                  step={1}
                  value={[config.learning_rate * 1000]}
                  onValueChange={(val) => {
                    const v = Array.isArray(val) ? val[0] : val;
                    setConfig({ ...config, learning_rate: v / 1000 });
                  }}
                />
              </div>
            </div>
            <div className="mt-6">
              <Button
                id="start-training-btn"
                className="w-full text-base font-semibold h-12 cursor-pointer"
                onClick={handleTrain}
                disabled={loading}
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <span className="h-4 w-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
                    Training…
                  </span>
                ) : (
                  "Start Training"
                )}
              </Button>
            </div>
            {error && (
              <p className="mt-4 text-destructive text-sm">Error: {error}</p>
            )}
          </CardContent>
        </Card>

        {/* Summary Stats */}
        <div className="space-y-4">
          <StatCard
            label="Final Accuracy"
            value={finalAcc != null ? `${(finalAcc * 100).toFixed(2)}%` : "—"}
          />
          <StatCard
            label="Final Loss"
            value={finalLoss != null ? finalLoss.toFixed(4) : "—"}
          />
          <StatCard
            label="Total Dropouts"
            value={totalDropped != null ? String(totalDropped) : "—"}
          />
        </div>
      </div>

      {/* Charts */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Accuracy Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Accuracy vs Rounds</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={result.rounds}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.3 0.02 265)" />
                  <XAxis dataKey="round" stroke="oklch(0.6 0.02 260)" fontSize={12} />
                  <YAxis stroke="oklch(0.6 0.02 260)" fontSize={12} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: "oklch(0.2 0.02 265)", border: "1px solid oklch(0.3 0.03 265)", borderRadius: 8 }} />
                  <Line type="monotone" dataKey="accuracy" stroke="oklch(0.65 0.2 260)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Loss Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Loss vs Rounds</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={result.rounds}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.3 0.02 265)" />
                  <XAxis dataKey="round" stroke="oklch(0.6 0.02 260)" fontSize={12} />
                  <YAxis stroke="oklch(0.6 0.02 260)" fontSize={12} />
                  <Tooltip contentStyle={{ background: "oklch(0.2 0.02 265)", border: "1px solid oklch(0.3 0.03 265)", borderRadius: 8 }} />
                  <Line type="monotone" dataKey="loss" stroke="oklch(0.7 0.2 40)" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Dropout Bar Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Client Activity per Round</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={result.rounds}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.3 0.02 265)" />
                  <XAxis dataKey="round" stroke="oklch(0.6 0.02 260)" fontSize={12} />
                  <YAxis stroke="oklch(0.6 0.02 260)" fontSize={12} />
                  <Tooltip contentStyle={{ background: "oklch(0.2 0.02 265)", border: "1px solid oklch(0.3 0.03 265)", borderRadius: 8 }} />
                  <Legend />
                  <Bar dataKey="active_count" name="Active" fill="oklch(0.7 0.18 150)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="dropped_count" name="Dropped" fill="oklch(0.6 0.22 25)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Client Reliability */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Client Reliability Scores</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={result.clients.slice(0, 20)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.3 0.02 265)" />
                  <XAxis dataKey="id" stroke="oklch(0.6 0.02 260)" fontSize={12} label={{ value: "Client ID", position: "insideBottom", offset: -2, fill: "oklch(0.6 0.02 260)" }} />
                  <YAxis stroke="oklch(0.6 0.02 260)" fontSize={12} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: "oklch(0.2 0.02 265)", border: "1px solid oklch(0.3 0.03 265)", borderRadius: 8 }} />
                  <Bar dataKey="reliability_score" name="Reliability" fill="oklch(0.65 0.22 330)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}
    </main>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card className="transition-smooth hover:scale-[1.02]">
      <CardContent className="pt-6">
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="text-3xl font-bold tracking-tight mt-1">{value}</p>
      </CardContent>
    </Card>
  );
}

"use client";

import { useState } from "react";
import {
	Activity,
	BrainCircuit,
	TrendingUp,
	Users,
	Package,
	ShoppingCart,
	Play,
	RefreshCw,
	Sparkles,
	BarChart3,
	Award
} from "lucide-react";
import { smartCartApi } from "@/lib/smartcart-api";
import { Button } from "@/ui/components/ui/button";

interface AnalyticsDashboardClientProps {
	initialOverview: any;
	initialComparison: any;
	locale: string;
	channel: string;
}

export function AnalyticsDashboardClient({ initialOverview, initialComparison }: AnalyticsDashboardClientProps) {
	const [overview, setOverview] = useState<any>(initialOverview);
	const [comparison, setComparison] = useState<any>(initialComparison);
	const [simSteps, setSimSteps] = useState(100);
	const [simSeed, setSimSeed] = useState(42);
	const [isRunningSim, setIsRunningSim] = useState(false);
	const [isRefreshing, setIsRefreshing] = useState(false);

	const handleRefresh = async () => {
		setIsRefreshing(true);
		try {
			const [ov, comp] = await Promise.all([
				smartCartApi.getAnalyticsOverview(),
				smartCartApi.getAlgorithmComparison(),
			]);
			setOverview(ov);
			setComparison(comp);
		} catch (e) {
			console.error("Refresh error:", e);
		} finally {
			setIsRefreshing(false);
		}
	};

	const handleRunSimulation = async (e: React.FormEvent) => {
		e.preventDefault();
		setIsRunningSim(true);
		try {
			const res = await smartCartApi.runSimulation(simSteps, simSeed);
			setComparison(res);
		} catch (e) {
			console.error("Simulation run error:", e);
		} finally {
			setIsRunningSim(false);
		}
	};

	const bandit = overview?.bandit || {};
	const compResults = comparison?.results || {};

	return (
		<div className="space-y-10">
			{/* Header */}
			<div className="flex flex-col justify-between gap-4 border-b pb-6 sm:flex-row sm:items-center">
				<div>
					<div className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
						<BrainCircuit className="h-3.5 w-3.5" />
						Research & Bandit Dashboard
					</div>
					<h1 className="mt-2 text-3xl font-extrabold tracking-tight text-foreground">
						SmartCart AI Analytics & Research Lab
					</h1>
					<p className="mt-1 text-sm text-muted-foreground">
						Real-time metrics for Contextual Bandit recommendations, online policy updates, and algorithm benchmarks.
					</p>
				</div>
				<Button
					variant="outline"
					size="sm"
					onClick={handleRefresh}
					disabled={isRefreshing}
					className="self-start sm:self-auto"
				>
					<RefreshCw className={`mr-2 h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} /> Refresh Data
				</Button>
			</div>

			{/* KPI Grid */}
			<div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
				<div className="rounded-xl border bg-card p-4 shadow-sm">
					<div className="flex items-center justify-between text-muted-foreground">
						<span className="text-xs font-medium">Catalog Products</span>
						<Package className="h-4 w-4" />
					</div>
					<p className="mt-2 text-2xl font-extrabold text-foreground">{overview?.total_products ?? 194}</p>
					<span className="text-[11px] text-emerald-600">MongoDB Atlas Active</span>
				</div>

				<div className="rounded-xl border bg-card p-4 shadow-sm">
					<div className="flex items-center justify-between text-muted-foreground">
						<span className="text-xs font-medium">Customer Events</span>
						<Activity className="h-4 w-4" />
					</div>
					<p className="mt-2 text-2xl font-extrabold text-foreground">{overview?.total_interactions ?? 0}</p>
					<span className="text-[11px] text-muted-foreground">Clicks, views, carts</span>
				</div>

				<div className="rounded-xl border bg-card p-4 shadow-sm">
					<div className="flex items-center justify-between text-muted-foreground">
						<span className="text-xs font-medium">Orders Placed</span>
						<ShoppingCart className="h-4 w-4" />
					</div>
					<p className="mt-2 text-2xl font-extrabold text-foreground">{overview?.total_orders ?? 0}</p>
					<span className="text-[11px] text-emerald-600">Validated purchases</span>
				</div>

				<div className="rounded-xl border bg-card p-4 shadow-sm">
					<div className="flex items-center justify-between text-muted-foreground">
						<span className="text-xs font-medium">LinUCB Updates</span>
						<Sparkles className="h-4 w-4 text-primary" />
					</div>
					<p className="mt-2 text-2xl font-extrabold text-primary">{bandit?.total_updates ?? 0}</p>
					<span className="text-[11px] text-muted-foreground">Ridge matrix steps</span>
				</div>

				<div className="rounded-xl border bg-card p-4 shadow-sm">
					<div className="flex items-center justify-between text-muted-foreground">
						<span className="text-xs font-medium">Cumulative Reward</span>
						<Award className="h-4 w-4 text-amber-500" />
					</div>
					<p className="mt-2 text-2xl font-extrabold text-foreground">{bandit?.cumulative_reward ?? 0.0}</p>
					<span className="text-[11px] text-muted-foreground">Avg: {bandit?.average_reward ?? 0.0} / step</span>
				</div>

				<div className="rounded-xl border bg-card p-4 shadow-sm">
					<div className="flex items-center justify-between text-muted-foreground">
						<span className="text-xs font-medium">Click-Through Rate</span>
						<TrendingUp className="h-4 w-4 text-blue-500" />
					</div>
					<p className="mt-2 text-2xl font-extrabold text-foreground">{overview?.click_through_rate ?? 0}%</p>
					<span className="text-[11px] text-muted-foreground">Conversion: {overview?.conversion_rate ?? 0}%</span>
				</div>
			</div>

			{/* LinUCB Agent Architecture Specification */}
			<div className="rounded-2xl border bg-gradient-to-r from-card via-card to-primary/5 p-6 shadow-sm">
				<h2 className="text-lg font-bold text-foreground">Active Bandit Algorithm Specification</h2>
				<div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-4 text-sm">
					<div className="rounded-lg border bg-background/50 p-3">
						<span className="text-xs text-muted-foreground">Algorithm</span>
						<p className="font-bold text-foreground">{bandit?.algorithm || "Contextual LinUCB"}</p>
					</div>
					<div className="rounded-lg border bg-background/50 p-3">
						<span className="text-xs text-muted-foreground">Exploration Parameter (α)</span>
						<p className="font-bold text-foreground">{bandit?.alpha ?? 0.8}</p>
					</div>
					<div className="rounded-lg border bg-background/50 p-3">
						<span className="text-xs text-muted-foreground">Feature Vector Dimension (d)</span>
						<p className="font-bold text-foreground">{bandit?.dimension ?? 8} dimensions</p>
					</div>
					<div className="rounded-lg border bg-background/50 p-3">
						<span className="text-xs text-muted-foreground">Active Category Arms</span>
						<p className="font-bold text-foreground">{bandit?.arms_count ?? 24} arms</p>
					</div>
				</div>
			</div>

			{/* Research Simulation & Algorithm Comparison */}
			<div className="rounded-2xl border bg-card p-6 shadow-sm">
				<div className="flex flex-col justify-between gap-4 border-b pb-4 lg:flex-row lg:items-center">
					<div>
						<h2 className="text-xl font-bold text-foreground">
							Algorithm Benchmark: Popularity vs. ε-Greedy vs. Contextual LinUCB
						</h2>
						<p className="text-xs text-muted-foreground">
							Evaluated across simulated heterogeneous customer populations (price sensitive, brand loyalty, category affinity).
						</p>
					</div>
					<form onSubmit={handleRunSimulation} className="flex items-center gap-2">
						<div className="flex items-center gap-1.5 text-xs text-muted-foreground">
							<span>Steps:</span>
							<input
								type="number"
								min="20"
								max="500"
								value={simSteps}
								onChange={(e) => setSimSteps(Number(e.target.value))}
								className="w-16 rounded border bg-background px-2 py-1 text-xs text-foreground"
							/>
						</div>
						<div className="flex items-center gap-1.5 text-xs text-muted-foreground">
							<span>Seed:</span>
							<input
								type="number"
								value={simSeed}
								onChange={(e) => setSimSeed(Number(e.target.value))}
								className="w-14 rounded border bg-background px-2 py-1 text-xs text-foreground"
							/>
						</div>
						<Button size="sm" type="submit" disabled={isRunningSim}>
							<Play className="mr-1 h-3.5 w-3.5" /> {isRunningSim ? "Simulating..." : "Run Experiment"}
						</Button>
					</form>
				</div>

				{/* Comparison Table */}
				<div className="mt-6 overflow-x-auto">
					<table className="w-full text-left text-sm">
						<thead>
							<tr className="border-b text-xs font-semibold text-muted-foreground">
								<th className="pb-3">Algorithm</th>
								<th className="pb-3">Total Impressions</th>
								<th className="pb-3">Cumulative Reward</th>
								<th className="pb-3">Avg Reward / Step</th>
								<th className="pb-3">CTR</th>
								<th className="pb-3">Conversion Rate</th>
								<th className="pb-3">Negative Feedback Rate</th>
								<th className="pb-3">Category Diversity</th>
							</tr>
						</thead>
						<tbody className="divide-y text-sm">
							{["popularity", "epsilon_greedy", "linucb"].map((key) => {
								const data = compResults[key];
								if (!data) return null;
								const isLinucb = key === "linucb";

								return (
									<tr key={key} className={isLinucb ? "bg-primary/5 font-semibold" : ""}>
										<td className="py-3.5">
											<div className="flex items-center gap-2">
												{isLinucb && <Sparkles className="h-4 w-4 text-primary" />}
												<span className={isLinucb ? "text-primary font-bold" : "text-foreground"}>
													{data.algorithm}
												</span>
											</div>
										</td>
										<td className="py-3.5">{data.total_impressions}</td>
										<td className="py-3.5">
											<span className={isLinucb ? "text-primary font-extrabold" : ""}>
												{data.cumulative_reward}
											</span>
										</td>
										<td className="py-3.5 font-mono">{data.average_reward}</td>
										<td className="py-3.5 font-mono">{(data.click_through_rate * 100).toFixed(1)}%</td>
										<td className="py-3.5 font-mono">{(data.conversion_rate * 100).toFixed(1)}%</td>
										<td className="py-3.5 font-mono text-red-600">{(data.negative_feedback_rate * 100).toFixed(1)}%</td>
										<td className="py-3.5 font-mono">{data.category_diversity} categories</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</div>

				{/* Reward Trajectory Visualization */}
				{compResults?.linucb?.cumulative_reward_curve && (
					<div className="mt-8 border-t pt-6">
						<h3 className="mb-4 text-sm font-bold text-foreground">
							Cumulative Reward Trajectory Over Simulation Steps
						</h3>
						<div className="grid grid-cols-1 gap-4 md:grid-cols-3">
							{["popularity", "epsilon_greedy", "linucb"].map((key) => {
								const data = compResults[key];
								if (!data?.cumulative_reward_curve) return null;
								const curve = data.cumulative_reward_curve;
								const maxVal = Math.max(...curve, 1);

								return (
									<div key={key} className="rounded-xl border bg-muted/20 p-4">
										<div className="flex justify-between text-xs font-semibold">
											<span>{data.algorithm}</span>
											<span className="font-mono text-primary">${data.cumulative_reward}</span>
										</div>
										<div className="mt-4 flex h-24 items-end gap-1">
											{curve.map((val: number, i: number) => {
												const heightPercent = Math.max(5, Math.min(100, (val / maxVal) * 100));
												return (
													<div
														key={i}
														style={{ height: `${heightPercent}%` }}
														className={`flex-1 rounded-t transition-all ${
															key === "linucb" ? "bg-primary" : "bg-muted-foreground/40"
														}`}
														title={`Step ${i * 5}: Reward ${val}`}
													/>
												);
											})}
										</div>
										<p className="mt-2 text-center text-[10px] text-muted-foreground">
											Step progression →
										</p>
									</div>
								);
							})}
						</div>
					</div>
				)}
			</div>

			{/* Recent Policy Updates Stream */}
			{bandit?.recent_history?.length > 0 && (
				<div className="rounded-2xl border bg-card p-6 shadow-sm">
					<h2 className="text-lg font-bold text-foreground">Recent Policy Updates (Reward Stream)</h2>
					<div className="mt-4 divide-y">
						{bandit.recent_history.map((entry: any, i: number) => {
							const isPositive = entry.reward > 0;
							return (
								<div key={i} className="flex items-center justify-between py-2.5 text-xs">
									<div className="flex items-center gap-2">
										<span className="font-mono text-muted-foreground">Step #{entry.step}</span>
										<span className="font-semibold text-foreground">Arm: {entry.arm_id}</span>
										<span className="rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">
											{entry.event_type}
										</span>
									</div>
									<div className="flex items-center gap-3">
										<span
											className={`font-mono font-bold ${
												isPositive ? "text-emerald-600" : "text-red-600"
											}`}
										>
											{isPositive ? `+${entry.reward}` : entry.reward}
										</span>
										<span className="text-[10px] text-muted-foreground">
											{new Date(entry.timestamp).toLocaleTimeString()}
										</span>
									</div>
								</div>
							);
						})}
					</div>
				</div>
			)}
		</div>
	);
}


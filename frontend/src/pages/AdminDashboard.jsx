import React, { useState, useEffect } from 'react';
import {
  BrainCircuit,
  TrendingUp,
  Activity,
  Layers,
  Settings,
  RotateCcw,
  Play,
  CheckCircle,
  Eye,
  Sliders,
  Sparkles,
  BarChart2,
  Clock,
  Shield,
  Archive,
  Database
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import { api } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';

export default function AdminDashboard() {
  const { isAdmin } = useAuth();
  const { addToast } = useToast();

  const [metrics, setMetrics] = useState(null);
  const [algorithmsData, setAlgorithmsData] = useState([]);
  const [activeAlgo, setActiveAlgo] = useState('linucb');
  const [rewardConfig, setRewardConfig] = useState(null);
  const [logs, setLogs] = useState([]);
  const [simRounds, setSimRounds] = useState(25);
  const [trainLive, setTrainLive] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [windowFilter, setWindowFilter] = useState('all');
  const [originFilter, setOriginFilter] = useState('live');
  const [loading, setLoading] = useState(true);

  // Poll metrics every 4 seconds
  const fetchMetrics = async () => {
    try {
      const [metRes, algoRes, rewRes, logRes] = await Promise.all([
        api.getMetrics(windowFilter, originFilter),
        api.getAlgorithms(),
        api.getRewardConfig(),
        api.getLogs(15, undefined, originFilter),
      ]);
      setMetrics(metRes.data);
      setAlgorithmsData(metRes.data?.algorithms || metRes.data?.metrics || []);
      setActiveAlgo(algoRes.data?.active_algorithm || 'linucb');
      setRewardConfig(rewRes.data);
      setLogs(logRes.data || []);
    } catch (e) {
      console.error('Failed to load admin metrics:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 4000);
    return () => clearInterval(interval);
  }, [windowFilter, originFilter]);

  const handleAlgorithmChange = async (newAlgo) => {
    try {
      await api.setAlgorithm(newAlgo);
      setActiveAlgo(newAlgo);
      addToast(`Active algorithm switched to ${newAlgo.toUpperCase()}`, 'success');
      await fetchMetrics();
    } catch (e) {
      addToast('Failed to change active algorithm.', 'error');
    }
  };

  const handleSimulate = async () => {
    setSimulating(true);
    try {
      const res = await api.runSimulate(simRounds, trainLive);
      const target = trainLive ? 'LIVE models' : 'SANDBOX (sim:*)';
      addToast(
        `Simulation finished: ${res.data.rounds_executed} rounds (${target}), ${res.data.total_events_generated} events created.`,
        'success'
      );
      await fetchMetrics();
    } catch (e) {
      addToast('Simulation run failed.', 'error');
    } finally {
      setSimulating(false);
    }
  };

  const handleResetModels = async () => {
    if (!window.confirm('Reset live bandit model weights back to priors?')) return;
    try {
      await api.resetModels(false);
      addToast('All live recommendation models reset to initial state.', 'info');
      await fetchMetrics();
    } catch (e) {
      addToast('Failed to reset models.', 'error');
    }
  };

  const handleSnapshot = async () => {
    const snapName = prompt('Enter snapshot label name:', 'pre_phase8');
    if (!snapName) return;
    try {
      const res = await api.createModelSnapshot(snapName);
      addToast(`Archived ${res.data.models_archived} model states to snapshot '${snapName}'.`, 'success');
    } catch (e) {
      addToast('Failed to create model snapshot.', 'error');
    }
  };

  const handleRollback = async () => {
    const snapName = prompt('Enter snapshot label name to restore:', 'pre_phase8');
    if (!snapName) return;
    try {
      await api.rollbackModelSnapshot(snapName);
      addToast(`Successfully rolled back to snapshot '${snapName}'.`, 'info');
      await fetchMetrics();
    } catch (e) {
      addToast(`Failed to restore snapshot '${snapName}'.`, 'error');
    }
  };

  const handleRewardUpdate = async (key, val) => {
    const updated = { ...rewardConfig, [key]: parseFloat(val) };
    setRewardConfig(updated);
    try {
      await api.updateRewardConfig(updated);
      addToast(`Reward for ${key} updated to ${val}`, 'success');
    } catch (e) {
      addToast('Failed to update reward config.', 'error');
    }
  };

  if (!isAdmin) {
    return (
      <div className="max-w-md mx-auto my-20 p-8 rounded-3xl bg-white dark:bg-slate-900 border text-center space-y-3">
        <div className="p-3 bg-rose-50 text-rose-600 rounded-full w-fit mx-auto">
          <BrainCircuit className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold">Admin Privileges Required</h2>
        <p className="text-xs text-slate-500">
          Please log in as an administrator (e.g. admin@smartcart.ai / admin123) to access the Live Bandit Monitoring Dashboard.
        </p>
      </div>
    );
  }

  // Pre-formatted chart data
  const chartData = algorithmsData.map((a) => ({
    name: a.algorithm.replace('_', ' ').toUpperCase(),
    rawName: a.algorithm,
    cumulativeReward: a.cumulative_reward,
    averageReward: a.average_reward,
    ctr: a.ctr,
    conversionRate: a.conversion_rate,
    negativeFeedbackRate: a.negative_feedback_rate,
    catalogueCoverage: a.catalogue_coverage,
    eventsCount: a.events_count,
    explorationPct: a.exploration_pct,
  }));

  const e2eP50 = metrics?.overall?.e2e_p50_ms || 0;
  const e2eP95 = metrics?.overall?.e2e_p95_ms || 0;
  const e2eP99 = metrics?.overall?.e2e_p99_ms || 0;
  const modelP50 = metrics?.overall?.model_scoring_p50_ms || 0;
  const modelP95 = metrics?.overall?.model_scoring_p95_ms || 0;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
      {/* Top Banner & Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-200 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white">
              Adaptive Bandit Monitoring Dashboard
            </h1>
            <span className="flex h-2.5 w-2.5 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Real-time comparison: LinUCB (Sherman-Morrison), Discounted LinUCB, LinTS, Epsilon-Greedy, Popularity, and Random baselines.
          </p>
        </div>

        {/* Global Controls: Origin & Time Window */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Origin Filter */}
          <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl text-xs font-semibold">
            {['live', 'simulated', 'legacy', 'all'].map((orig) => (
              <button
                key={orig}
                onClick={() => setOriginFilter(orig)}
                className={`px-2.5 py-1 rounded-lg capitalize transition ${
                  originFilter === orig
                    ? 'bg-white dark:bg-slate-900 text-indigo-600 shadow-sm font-bold'
                    : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                {orig}
              </button>
            ))}
          </div>

          {/* Time Window */}
          <select
            value={windowFilter}
            onChange={(e) => setWindowFilter(e.target.value)}
            className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
          >
            <option value="all">All Time History</option>
            <option value="1h">Last 1 Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
          </select>

          {/* Snapshot Controls */}
          <button
            onClick={handleSnapshot}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-700 text-xs font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition"
            title="Archive a versioned snapshot of all model weights"
          >
            <Archive className="w-3.5 h-3.5" />
            <span>Snapshot</span>
          </button>

          <button
            onClick={handleRollback}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-700 text-xs font-bold hover:bg-slate-100 dark:hover:bg-slate-800 transition"
            title="Rollback model weights from an archived snapshot"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Rollback</span>
          </button>
        </div>
      </div>

      {/* KPI Overview Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-xs font-semibold">Active Catalog</span>
            <Database className="w-4 h-4 text-indigo-500" />
          </div>
          <div className="text-2xl font-black text-slate-900 dark:text-white">
            {metrics?.active_catalog_products || 262}
          </div>
          <span className="text-[11px] text-slate-400">
            of {metrics?.total_catalog_products || 787} total catalog items
          </span>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-xs font-semibold">Tracked Events ({originFilter})</span>
            <Activity className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-2xl font-black text-slate-900 dark:text-white">
            {metrics?.overall?.total_events || 0}
          </div>
          <span className="text-[11px] text-slate-400">
            Avg Reward: {metrics?.overall?.overall_avg_reward > 0 ? `+${metrics?.overall?.overall_avg_reward}` : metrics?.overall?.overall_avg_reward || 0}
          </span>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-xs font-semibold">End-to-End Latency</span>
            <Clock className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-xl font-black text-slate-900 dark:text-white">
            {e2eP50} <span className="text-xs font-normal text-slate-400">ms (p50)</span>
          </div>
          <span className="text-[11px] text-slate-400">
            p95: {e2eP95} ms • p99: {e2eP99} ms
          </span>
        </div>

        <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-xs font-semibold">Bandit Scoring Latency</span>
            <BrainCircuit className="w-4 h-4 text-purple-500" />
          </div>
          <div className="text-xl font-black text-slate-900 dark:text-white">
            {modelP50} <span className="text-xs font-normal text-slate-400">ms (p50)</span>
          </div>
          <span className="text-[11px] text-slate-400">
            p95: {modelP95} ms • Sherman-Morrison O(d²)
          </span>
        </div>
      </div>

      {/* Algorithm Selector & Simulation Bar */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Active Algorithm Switcher */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-500" />
              <span>Active Recommendation Policy</span>
            </h3>
            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
              Current: {activeAlgo.toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
            {[
              { id: 'linucb', label: 'LinUCB' },
              { id: 'discounted_linucb', label: 'Discounted LinUCB' },
              { id: 'linucb_thompson', label: 'LinTS' },
              { id: 'epsilon_greedy', label: 'Epsilon-Greedy' },
              { id: 'popularity', label: 'Popularity' },
              { id: 'random', label: 'Random' },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => handleAlgorithmChange(item.id)}
                className={`py-2 px-3 rounded-xl font-bold transition text-center truncate ${
                  activeAlgo === item.id
                    ? 'bg-indigo-600 text-white shadow-md'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Traffic Simulator with Sandbox Isolation */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Play className="w-4 h-4 text-emerald-500" />
              <span>Traffic Simulator (Multi-Cohort)</span>
            </h3>
            <div className="flex items-center gap-2 text-xs">
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                trainLive ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300' : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
              }`}>
                {trainLive ? 'Live Training' : 'Sandbox Isolated (sim:*)'}
              </span>
            </div>
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Inject synthetic multi-cohort customer traffic. By default, updates only sandbox models (sim:*) and writes origin=simulated data.
          </p>

          <div className="flex items-center gap-3 pt-2">
            <input
              type="number"
              min="5"
              max="200"
              value={simRounds}
              onChange={(e) => setSimRounds(parseInt(e.target.value) || 25)}
              className="w-20 px-3 py-2 rounded-xl text-xs font-bold bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-center"
            />
            <button
              onClick={handleSimulate}
              disabled={simulating}
              className={`flex-1 py-2 px-4 rounded-xl text-xs font-bold flex items-center justify-center gap-2 text-white shadow transition ${
                simulating
                  ? 'bg-slate-400 cursor-not-allowed'
                  : 'bg-emerald-600 hover:bg-emerald-500 active:scale-95'
              }`}
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{simulating ? 'Running Simulator...' : `Run ${simRounds} Rounds`}</span>
            </button>
            <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer">
              <input
                type="checkbox"
                checked={trainLive}
                onChange={(e) => setTrainLive(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Train Live</span>
            </label>
          </div>
        </div>
      </div>

      {/* Visualizations: Cumulative Reward & CTR */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cumulative Reward Chart */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-indigo-500" />
              <span>Cumulative Reward Payoff by Policy</span>
            </h3>
            <span className="text-xs text-slate-400">Higher is better</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                <XAxis dataKey="name" textAnchor="end" interval={0} angle={-25} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="cumulativeReward" fill="#6366f1" radius={[8, 8, 0, 0]} name="Cumulative Reward" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* CTR & Conversion Rates */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-500" />
              <span>Click-Through Rate (%) & Conversion Rate (%)</span>
            </h3>
            <span className="text-xs text-slate-400">Efficiency</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                <XAxis dataKey="name" textAnchor="end" interval={0} angle={-25} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} unit="%" />
                <Tooltip />
                <Bar dataKey="ctr" fill="#10b981" radius={[8, 8, 0, 0]} name="CTR %" />
                <Bar dataKey="conversionRate" fill="#8b5cf6" radius={[8, 8, 0, 0]} name="Conversion %" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Reward Weight Configuration Editor */}
      {rewardConfig && (
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Settings className="w-4 h-4 text-indigo-500" />
              <span>Reward Matrix Configuration</span>
            </h3>
            <span className="text-xs text-slate-400">Real-time online gradient updates</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 text-xs">
            {Object.entries(rewardConfig).map(([key, val]) => (
              <div key={key} className="p-3 rounded-2xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400 truncate block">
                  {key.replace('_', ' ')}
                </span>
                <input
                  type="number"
                  step="0.5"
                  value={val}
                  onChange={(e) => handleRewardUpdate(key, e.target.value)}
                  className="w-full text-center font-extrabold text-sm bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg py-1"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Real-Time Recommendation Audit Logs */}
      <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Eye className="w-4 h-4 text-purple-500" />
            <span>Recent Recommendation Audit Logs (Origin: {originFilter})</span>
          </h3>
          <span className="text-xs text-slate-400">Latest 15 scoring events</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-100 dark:border-slate-800 text-slate-400 font-semibold">
                <th className="py-2 px-3">Product</th>
                <th className="py-2 px-3">Category</th>
                <th className="py-2 px-3">Algorithm</th>
                <th className="py-2 px-3">Origin</th>
                <th className="py-2 px-3">Score</th>
                <th className="py-2 px-3">UCB Bonus</th>
                <th className="py-2 px-3">Mode</th>
                <th className="py-2 px-3">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {logs.map((log) => (
                <tr key={log.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition">
                  <td className="py-2 px-3 font-semibold text-slate-800 dark:text-slate-200 max-w-xs truncate">
                    {log.product_name}
                  </td>
                  <td className="py-2 px-3 text-slate-500">{log.category}</td>
                  <td className="py-2 px-3 font-mono font-bold text-indigo-600 dark:text-indigo-400">
                    {log.algorithm}
                  </td>
                  <td className="py-2 px-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold capitalize ${
                      log.origin === 'live'
                        ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                        : log.origin === 'simulated'
                        ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300'
                        : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300'
                    }`}>
                      {log.origin}
                    </span>
                  </td>
                  <td className="py-2 px-3 font-mono">{log.score}</td>
                  <td className="py-2 px-3 font-mono text-emerald-600">+{log.ucb_bonus}</td>
                  <td className="py-2 px-3">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                      log.was_exploration
                        ? 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300'
                        : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400'
                    }`}>
                      {log.was_exploration ? 'EXPLORE' : 'EXPLOIT'}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-slate-400">{new Date(log.created_at).toLocaleTimeString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

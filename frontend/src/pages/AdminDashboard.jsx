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
  BarChart2
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
  const [simulating, setSimulating] = useState(false);
  const [windowFilter, setWindowFilter] = useState('all');
  const [loading, setLoading] = useState(true);

  // Poll metrics every 4 seconds
  const fetchMetrics = async () => {
    try {
      const [metRes, algoRes, rewRes, logRes] = await Promise.all([
        api.getMetrics(windowFilter),
        api.getAlgorithms(),
        api.getRewardConfig(),
        api.getLogs(15),
      ]);
      setMetrics(metRes.data);
      setAlgorithmsData(metRes.data?.algorithms || []);
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
  }, [windowFilter]);

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
      const res = await api.runSimulate(simRounds);
      addToast(
        `Simulation finished: ${res.data.rounds_executed} rounds, ${res.data.total_events_generated} events created.`,
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
    if (!window.confirm('Reset all bandit model weights and parameters back to priors?')) return;
    try {
      await api.resetModels();
      addToast('All recommendation models reset to initial state.', 'info');
      await fetchMetrics();
    } catch (e) {
      addToast('Failed to reset models.', 'error');
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
    explorationRatio: a.exploration_ratio,
  }));

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
            Real-time live comparison: LinUCB (Sherman-Morrison), Discounted LinUCB, LinTS, Epsilon-Greedy, Popularity, and Random baselines.
          </p>
        </div>

        {/* Global Controls */}
        <div className="flex flex-wrap items-center gap-3">
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

          {/* Reset Models Button */}
          <button
            onClick={handleResetModels}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-rose-200 bg-rose-50 dark:bg-rose-950/50 text-rose-700 dark:text-rose-300 text-xs font-bold hover:bg-rose-100 transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Models</span>
          </button>
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
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Traffic Simulator Controller */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <Play className="w-4 h-4 text-emerald-500" />
              <span>Simulated Shopping Traffic Generator</span>
            </h3>
            <span className="text-xs text-slate-400">Populate live curves</span>
          </div>

          <div className="flex items-center gap-3 pt-1">
            <div className="flex items-center gap-2 text-xs font-medium">
              <span>Rounds:</span>
              <input
                type="number"
                min="5"
                max="250"
                value={simRounds}
                onChange={(e) => setSimRounds(parseInt(e.target.value, 10) || 25)}
                className="w-20 px-2.5 py-1.5 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-center"
              />
            </div>

            <button
              onClick={handleSimulate}
              disabled={simulating}
              className="flex-1 py-2 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-md transition disabled:opacity-50 flex items-center justify-center gap-2"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{simulating ? 'Generating Traffic...' : 'Run Simulation Batch'}</span>
            </button>
          </div>
          <p className="text-[11px] text-slate-400">
            Executes multi-persona customer shopping sessions to benchmark bandit adaptation rates.
          </p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* 1. Cumulative Reward Chart */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-indigo-500" />
            <span>Cumulative Reward by Policy (Higher is Better)</span>
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="name" textAnchor="end" height={60} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="cumulativeReward" name="Cumulative Reward" fill="#4f46e5" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 2. CTR & Conversion Rate Chart */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Activity className="w-4 h-4 text-emerald-500" />
            <span>Click-Through Rate (%) & Conversion Rate (%)</span>
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="name" textAnchor="end" height={60} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} unit="%" />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="ctr" name="CTR (%)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="conversionRate" name="Conv Rate (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 3. Negative Feedback Rate vs Exploration */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-amber-500" />
            <span>Negative Feedback Rate (%) vs Exploration Ratio (%)</span>
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="name" textAnchor="end" height={60} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} unit="%" />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="negativeFeedbackRate" name="Dislikes + Returns (%)" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                <Bar dataKey="explorationRatio" name="Exploration Ratio (%)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. Catalogue Coverage % */}
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-purple-500" />
            <span>Catalogue Exploration Coverage (% of 630 Products)</span>
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="name" textAnchor="end" height={60} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} unit="%" domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="catalogueCoverage" name="Catalogue Coverage (%)" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Dynamic Reward Config Editor */}
      {rewardConfig && (
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Settings className="w-4 h-4 text-indigo-500" />
            <span>Dynamic Reward Function Weights</span>
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            {Object.entries(rewardConfig).map(([k, val]) => (
              <div key={k} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-xl space-y-1">
                <span className="font-semibold text-slate-600 dark:text-slate-400 capitalize">{k.replace('_', ' ')}</span>
                <input
                  type="number"
                  step="0.5"
                  value={val}
                  onChange={(e) => handleRewardUpdate(k, e.target.value)}
                  className="w-full px-2 py-1 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded text-center font-bold"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Live Recommendation Decision Logs */}
      <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-4">
        <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Eye className="w-4 h-4 text-indigo-500" />
          <span>Latest Bandit Decisions (Live Audit Trail)</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 uppercase text-[10px]">
                <th className="py-2.5 px-3">Product</th>
                <th className="py-2.5 px-3">Category</th>
                <th className="py-2.5 px-3">Price</th>
                <th className="py-2.5 px-3">Algorithm</th>
                <th className="py-2.5 px-3">Score</th>
                <th className="py-2.5 px-3">UCB Bonus</th>
                <th className="py-2.5 px-3">Exploration?</th>
                <th className="py-2.5 px-3">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {logs.map((lg) => (
                <tr key={lg.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                  <td className="py-2 px-3 font-semibold text-slate-900 dark:text-white max-w-xs truncate">
                    {lg.product_name}
                  </td>
                  <td className="py-2 px-3 text-slate-500">{lg.category}</td>
                  <td className="py-2 px-3 font-medium">₹{lg.price.toLocaleString('en-IN')}</td>
                  <td className="py-2 px-3">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
                      {lg.algorithm}
                    </span>
                  </td>
                  <td className="py-2 px-3 font-bold">{lg.score}</td>
                  <td className="py-2 px-3 text-slate-500">{lg.ucb_bonus}</td>
                  <td className="py-2 px-3">
                    {lg.was_exploration ? (
                      <span className="text-amber-500 font-bold text-[10px]">YES</span>
                    ) : (
                      <span className="text-slate-400 text-[10px]">NO</span>
                    )}
                  </td>
                  <td className="py-2 px-3 text-slate-400 text-[10px]">
                    {new Date(lg.created_at).toLocaleTimeString('en-IN')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}


import React, { useState, useEffect } from 'react';
import { Sparkles, BrainCircuit, RefreshCw } from 'lucide-react';
import ProductCard from './ProductCard';
import { api } from '../api/client';

export default function RecommendationRow({ category = null, maxPrice = null, title = "Recommended For You" }) {
  const [recommendations, setRecommendations] = useState([]);
  const [algorithm, setAlgorithm] = useState('');
  const [coldStart, setColdStart] = useState(false);
  const [latency, setLatency] = useState(0);
  const [loading, setLoading] = useState(true);

  const fetchRecs = async () => {
    setLoading(true);
    const sessionId = localStorage.getItem('smartcart_session_id');
    try {
      const res = await api.getRecommendations({
        limit: 8,
        category: category || undefined,
        max_price: maxPrice || undefined,
        session_id: sessionId,
      });

      const recs = res.data.recommendations || [];
      setRecommendations(recs);
      setAlgorithm(res.data.algorithm || 'linucb');
      setColdStart(res.data.cold_start || false);
      setLatency(res.data.latency_ms || 0);

      // Record impressions
      if (recs.length > 0) {
        const pIds = recs.map((r) => r.product.id);
        const logIds = recs.map((r) => r.log_id).filter(Boolean);
        api.recordImpressions(pIds, logIds, sessionId).catch(() => {});
      }
    } catch (e) {
      console.error('Failed to load recommendations:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecs();
  }, [category, maxPrice]);

  const handleDislikeItem = (productId) => {
    setRecommendations((prev) => prev.filter((r) => r.product.id !== productId));
  };

  if (loading) {
    return (
      <section className="my-10">
        <div className="flex items-center justify-between mb-6">
          <div className="h-7 w-48 bg-slate-200 dark:bg-slate-800 rounded-lg animate-pulse" />
          <div className="h-6 w-32 bg-slate-200 dark:bg-slate-800 rounded-full animate-pulse" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-80 rounded-2xl bg-slate-100 dark:bg-slate-800 animate-pulse border border-slate-200 dark:border-slate-800" />
          ))}
        </div>
      </section>
    );
  }

  if (recommendations.length === 0) return null;

  const algoLabels = {
    linucb: 'LinUCB Contextual Bandit',
    discounted_linucb: 'Discounted LinUCB (Drift-Aware)',
    linucb_thompson: 'Linear Thompson Sampling',
    epsilon_greedy: 'Epsilon-Greedy Bandit',
    popularity: 'Popularity Prior Baseline',
    random: 'Random Policy',
  };

  return (
    <section className="my-12">
      {/* Header with Adaptive AI Badge */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white">
              {title}
            </h2>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-800 dark:bg-indigo-950 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800">
              <BrainCircuit className="w-3.5 h-3.5" />
              <span>{algoLabels[algorithm] || algorithm.toUpperCase()}</span>
            </div>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Real-time multi-armed bandit predictions • latency: {latency.toFixed(1)} ms
            {coldStart && ' • Cold-start priors active'}
          </p>
        </div>

        <button
          onClick={fetchRecs}
          className="flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-indigo-600 transition px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:border-indigo-300"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh AI Feed</span>
        </button>
      </div>

      {/* Grid of Recommended Products */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {recommendations.map((item) => (
          <ProductCard
            key={item.product.id}
            product={item.product}
            explanation={item.explanation}
            score={item.score}
            was_exploration={item.was_exploration}
            log_id={item.log_id}
            onDislike={handleDislikeItem}
          />
        ))}
      </div>
    </section>
  );
}


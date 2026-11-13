import React, { useState, useEffect } from 'react';
import { History, Eye } from 'lucide-react';
import ProductCard from './ProductCard';
import { api } from '../api/client';
import { useAuth } from '../context/AuthContext';

export default function RecentlyViewedRow({ title = "Recently Viewed" }) {
  const { user } = useAuth();
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);

  // Hidden when consent is declined
  if (user && user.consent_tracking === false) {
    return null;
  }

  const fetchRecentlyViewed = async () => {
    setLoading(true);
    const sessionId = localStorage.getItem('smartcart_session_id');
    try {
      const res = await api.getRecentlyViewed(8, sessionId);
      setProducts(res.data || []);
    } catch (e) {
      console.error('Failed to load recently viewed:', e);
      setProducts([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecentlyViewed();
  }, [user]);

  if (loading || products.length === 0) return null;

  return (
    <section className="my-12">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-lg">
            <History className="w-4 h-4" />
          </div>
          <h2 className="text-xl sm:text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white">
            {title}
          </h2>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            • Browsing history
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {products.map((prod) => (
          <ProductCard
            key={prod.id}
            product={prod}
            explanation="From your recent browsing history"
          />
        ))}
      </div>
    </section>
  );
}

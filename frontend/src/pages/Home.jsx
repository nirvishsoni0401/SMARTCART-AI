import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  Zap,
  Star,
  ArrowRight,
  TrendingUp,
  Cpu,
  Shirt,
  Utensils,
  Sparkle,
  Dumbbell,
  BookOpen
} from 'lucide-react';
import ProductCard from '../components/ProductCard';
import RecommendationRow from '../components/RecommendationRow';
import RecentlyViewedRow from '../components/RecentlyViewedRow';
import { api } from '../api/client';

export default function Home() {
  const [deals, setDeals] = useState([]);
  const [topRated, setTopRated] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadFeatured() {
      try {
        const [dealsRes, topRes] = await Promise.all([
          api.getProducts({ min_discount: 30, page_size: 4, sort_by: 'popularity' }),
          api.getProducts({ min_rating: 4.5, page_size: 4, sort_by: 'rating_desc' }),
        ]);
        setDeals(dealsRes.data.products || []);
        setTopRated(topRes.data.products || []);
      } catch (e) {
        console.error('Error fetching homepage products:', e);
      } finally {
        setLoading(false);
      }
    }
    loadFeatured();
  }, []);

  const categoryIcons = {
    Electronics: <Cpu className="w-6 h-6 text-blue-500" />,
    Fashion: <Shirt className="w-6 h-6 text-pink-500" />,
    'Home & Kitchen': <Utensils className="w-6 h-6 text-amber-500" />,
    Beauty: <Sparkle className="w-6 h-6 text-purple-500" />,
    Sports: <Dumbbell className="w-6 h-6 text-emerald-500" />,
    Books: <BookOpen className="w-6 h-6 text-indigo-500" />,
  };

  const categories = [
    { name: 'Electronics', count: '105+ Items', desc: 'Phones, Audio, Wearables' },
    { name: 'Fashion', count: '105+ Items', desc: 'Apparel, Footwear, Watches' },
    { name: 'Home & Kitchen', count: '105+ Items', desc: 'Cookware, Bedding, Decor' },
    { name: 'Beauty', count: '105+ Items', desc: 'Skincare, Fragrance, Haircare' },
    { name: 'Sports', count: '105+ Items', desc: 'Fitness, Gear, Yoga, Cricket' },
    { name: 'Books', count: '105+ Items', desc: 'Tech, Business, Fiction' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-12">
      {/* Hero Section */}
      <div className="relative rounded-3xl overflow-hidden bg-gradient-to-r from-indigo-900 via-indigo-800 to-purple-900 text-white p-8 sm:p-12 shadow-xl">
        <div className="max-w-2xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 text-indigo-200 text-xs font-bold tracking-wide backdrop-blur-md border border-white/10">
            <Sparkles className="w-3.5 h-3.5 text-amber-300" />
            600+ Products • Adaptive LinUCB Bandits
          </div>
          <h1 className="text-3xl sm:text-5xl font-black tracking-tight leading-tight">
            Next-Gen E-Commerce Powered by <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-300 to-pink-300">Reinforcement Learning</span>
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            Experience real-time personalization that continuously tunes recommendations to your clicks, cart additions, budget, and feedback.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <Link
              to="/products"
              className="px-6 py-3 rounded-xl bg-white text-indigo-950 font-bold text-sm shadow-lg hover:bg-slate-100 transition flex items-center gap-2"
            >
              <span>Explore 600+ Products</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/admin"
              className="px-5 py-3 rounded-xl bg-indigo-700/80 hover:bg-indigo-700 text-white font-semibold text-sm border border-indigo-400/30 transition flex items-center gap-2"
            >
              <span>Bandit Live Dashboard</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Shop by Department Grid */}
      <section>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white">
            Explore 6 Indian Departments
          </h2>
          <Link to="/products" className="text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:underline">
            View All →
          </Link>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {categories.map((cat) => (
            <Link
              key={cat.name}
              to={`/products?category=${encodeURIComponent(cat.name)}`}
              className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-indigo-500 dark:hover:border-indigo-500 hover:shadow-md transition text-center group"
            >
              <div className="p-3 mx-auto w-fit rounded-xl bg-slate-50 dark:bg-slate-800 group-hover:scale-110 transition mb-2">
                {categoryIcons[cat.name]}
              </div>
              <h3 className="text-xs font-bold text-slate-900 dark:text-white truncate">
                {cat.name}
              </h3>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">
                {cat.count}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* Recommended For You Section (LinUCB Contextual Bandits) */}
      <RecommendationRow title="Adaptive AI Picks For You" />

      {/* Recently Viewed Products (User History, Non-Bandit) */}
      <RecentlyViewedRow title="Recently Viewed" />

      {/* Today's Flash Deals */}
      {deals.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <div className="p-1.5 bg-rose-500 text-white rounded-lg">
                <Zap className="w-4 h-4 fill-current" />
              </div>
              <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white">
                Mega Deals (30% to 60% OFF)
              </h2>
            </div>
            <Link
              to="/products?min_discount=30&sort_by=price_asc"
              className="text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              See All Deals →
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {deals.map((p) => (
              <ProductCard key={p.id} product={p} />
            ))}
          </div>
        </section>
      )}

      {/* Top Rated Essentials */}
      {topRated.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <div className="p-1.5 bg-amber-500 text-white rounded-lg">
                <Star className="w-4 h-4 fill-current" />
              </div>
              <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white">
                Top Rated by Verified Buyers (4.5★+)
              </h2>
            </div>
            <Link
              to="/products?min_rating=4.5&sort_by=rating_desc"
              className="text-xs font-bold text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              See Top Rated →
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {topRated.map((p) => (
              <ProductCard key={p.id} product={p} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

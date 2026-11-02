import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, BrainCircuit, ShieldCheck, HeartHandshake } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-slate-900 text-slate-400 text-xs border-t border-slate-800 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Info */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-white">
              <div className="p-1.5 bg-indigo-600 rounded-lg">
                <Sparkles className="w-4 h-4" />
              </div>
              <span className="text-base font-bold tracking-tight">SmartCart AI</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Adaptive contextual bandit multi-category marketplace. Learning live from your preferences with Indian market brands and ₹ pricing.
            </p>
            <div className="flex items-center gap-2 text-indigo-400 font-semibold">
              <BrainCircuit className="w-4 h-4" />
              <span>LinUCB Online Learning Engine</span>
            </div>
          </div>

          {/* Quick Categories */}
          <div>
            <h4 className="text-white font-bold text-sm mb-3">Categories</h4>
            <ul className="space-y-2">
              {['Electronics', 'Fashion', 'Home & Kitchen', 'Beauty', 'Sports', 'Books'].map((cat) => (
                <li key={cat}>
                  <Link to={`/products?category=${encodeURIComponent(cat)}`} className="hover:text-white transition">
                    {cat}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Customer Support & Policies */}
          <div>
            <h4 className="text-white font-bold text-sm mb-3">Customer Service</h4>
            <ul className="space-y-2">
              <li><Link to="/orders" className="hover:text-white transition">Track & Return Orders</Link></li>
              <li><Link to="/cart" className="hover:text-white transition">Shopping Cart</Link></li>
              <li><Link to="/wishlist" className="hover:text-white transition">Your Wishlist</Link></li>
              <li><span className="text-slate-500">7-Day Free Replacement</span></li>
              <li><span className="text-slate-500">Secure UPI & Card Payments</span></li>
            </ul>
          </div>

          {/* Research & Bandits Info */}
          <div>
            <h4 className="text-white font-bold text-sm mb-3">AI Research Stack</h4>
            <div className="space-y-2 text-slate-400">
              <p>• LinUCB Sherman-Morrison rank-1</p>
              <p>• Discounted LinUCB for taste drift</p>
              <p>• Linear Thompson Sampling (LinTS)</p>
              <p>• 36-dimensional joint context phi(u, p)</p>
              <p>• MMR-style diversity re-ranking</p>
              <p>• Transparent explanation generation</p>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-800 mt-10 pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] text-slate-500">
          <p>© 2026 SmartCart AI. All rights reserved. Prices quoted in Indian Rupees (₹).</p>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" /> User Privacy Protected
            </span>
            <span className="flex items-center gap-1">
              <HeartHandshake className="w-3.5 h-3.5 text-indigo-400" /> Reinforcement Learning in Production
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}


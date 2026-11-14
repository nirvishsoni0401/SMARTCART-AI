import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Package, RotateCcw, CheckCircle, Clock, AlertCircle } from 'lucide-react';
import { api } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import ProductImage from '../components/ProductImage';

export default function OrderHistoryPage() {
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [returningItemId, setReturningItemId] = useState(null);
  const { user } = useAuth();
  const { addToast } = useToast();

  const fetchOrders = async () => {
    setLoading(true);
    try {
      const res = await api.getOrders();
      setOrders(res.data || []);
    } catch (e) {
      console.error('Failed to load orders:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOrders();
  }, []);

  const handleReturnItem = async (orderId, itemId) => {
    try {
      await api.returnItem(orderId, itemId, 'Customer initiated 7-day return');
      addToast('Item returned successfully. Refund initiated.', 'success');
      await fetchOrders();
    } catch (err) {
      const msg = err.response?.data?.detail || 'Return failed.';
      addToast(msg, 'error');
    } finally {
      setReturningItemId(null);
    }
  };

  if (!user) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center space-y-4">
        <Package className="w-12 h-12 mx-auto text-slate-400" />
        <h2 className="text-xl font-bold">Please log in to view your orders.</h2>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-12 space-y-6 animate-pulse">
        {[1, 2].map((i) => (
          <div key={i} className="h-44 bg-slate-100 dark:bg-slate-800 rounded-3xl" />
        ))}
      </div>
    );
  }

  if (orders.length === 0) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center space-y-4">
        <Package className="w-12 h-12 mx-auto text-slate-400" />
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">No Orders Placed Yet</h2>
        <p className="text-xs text-slate-500">
          When you place orders, they will appear here along with return management and delivery tracking.
        </p>
        <Link
          to="/products"
          className="inline-block mt-2 px-6 py-2.5 rounded-xl bg-indigo-600 text-white text-xs font-bold shadow-md hover:bg-indigo-500 transition"
        >
          Explore Catalog
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-black text-slate-900 dark:text-white">My Orders & Returns</h1>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Review purchase history and initiate returns. Returns emit delayed reward updates (-8.0) to your bandit policy.
        </p>
      </div>

      <div className="space-y-6">
        {orders.map((order) => (
          <div
            key={order.id}
            className="rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden shadow-sm"
          >
            {/* Order Header */}
            <div className="p-4 sm:p-6 bg-slate-50 dark:bg-slate-800/40 border-b border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs">
              <div>
                <span className="text-slate-400">Order ID:</span>{' '}
                <span className="font-bold text-slate-900 dark:text-white">#{order.id}</span>
                <span className="text-slate-300 mx-2">•</span>
                <span className="text-slate-500">
                  {new Date(order.created_at).toLocaleDateString('en-IN', {
                    day: 'numeric',
                    month: 'short',
                    year: 'numeric',
                  })}
                </span>
              </div>

              <div className="flex items-center gap-4">
                <span className="font-bold text-slate-900 dark:text-white">
                  Total: ₹{order.total_amount.toLocaleString('en-IN')}
                </span>
                <span className="px-2.5 py-1 rounded-full text-[11px] font-bold bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300 capitalize">
                  {order.status}
                </span>
              </div>
            </div>

            {/* Items */}
            <div className="p-4 sm:p-6 divide-y divide-slate-100 dark:divide-slate-800">
              {order.items.map((item) => (
                <div key={item.id} className="py-4 first:pt-0 last:pb-0 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className="w-16 h-16 rounded-xl overflow-hidden shrink-0">
                      <ProductImage
                        product={item.product}
                        aspectRatio="aspect-square"
                        hoverZoom={false}
                      />
                    </div>
                    <div>
                      <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600">
                        {item.product.brand} • {item.product.category}
                      </span>
                      <h4 className="text-sm font-semibold text-slate-900 dark:text-white line-clamp-1">
                        <Link to={`/products/${item.product.id}`} className="hover:underline">
                          {item.product.name}
                        </Link>
                      </h4>
                      <div className="text-xs text-slate-500 mt-0.5">
                        Qty: {item.quantity} • ₹{item.price.toLocaleString('en-IN')} each
                      </div>
                    </div>
                  </div>

                  {/* Return Item Action */}
                  <div className="flex items-center gap-3 shrink-0">
                    {item.status === 'returned' ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 dark:bg-rose-950 dark:text-rose-300">
                        <CheckCircle className="w-3.5 h-3.5" /> Returned
                      </span>
                    ) : (
                      <button
                        onClick={() => handleReturnItem(order.id, item.id)}
                        className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border border-slate-200 dark:border-slate-700 text-xs font-semibold text-slate-700 dark:text-slate-300 hover:text-rose-600 hover:border-rose-300 transition"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        <span>Return Item</span>
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


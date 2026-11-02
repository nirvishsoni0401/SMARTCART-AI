import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Star, Heart, ShoppingBag, ThumbsDown, Sparkles } from 'lucide-react';
import { useCartWishlist } from '../context/CartWishlistContext';
import { api } from '../api/client';

export default function ProductCard({
  product,
  explanation = null,
  score = null,
  was_exploration = false,
  log_id = null,
  onDislike = null,
}) {
  const { addToCart, addToWishlist, removeFromWishlist, inWishlist } = useCartWishlist();
  const [imgError, setImgError] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [adding, setAdding] = useState(false);

  if (!product || dismissed) return null;

  const isFav = inWishlist(product.id);

  const handleDislike = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDismissed(true);

    const sessionId = localStorage.getItem('smartcart_session_id');
    try {
      await api.recordEvent(product.id, 'dislike', log_id, sessionId);
    } catch (err) {
      console.error('Failed to log dislike event:', err);
    }

    if (onDislike) {
      onDislike(product.id);
    }
  };

  const handleWishlistToggle = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (isFav) {
      await removeFromWishlist(product.id);
    } else {
      await addToWishlist(product.id);
    }
  };

  const handleAddToCart = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setAdding(true);
    await addToCart(product.id, 1);
    setTimeout(() => setAdding(false), 800);
  };

  const handleCardClick = () => {
    const sessionId = localStorage.getItem('smartcart_session_id');
    api.recordEvent(product.id, 'click', log_id, sessionId).catch(() => {});
  };

  // Fallback placeholder SVG
  const fallbackSvg = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400" fill="%23f1f5f9"><rect width="400" height="400" fill="%23f8fafc"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="20" fill="%2394a3b8">${encodeURIComponent(product.category || 'Product')}</text></svg>`;

  return (
    <div className="group relative flex flex-col justify-between bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm hover:shadow-xl hover:border-indigo-400 dark:hover:border-indigo-500 transition-all duration-300">
      <div>
        {/* Top Badges & Actions */}
        <div className="relative aspect-square w-full overflow-hidden rounded-xl bg-slate-100 dark:bg-slate-800 mb-3">
          <Link to={`/products/${product.id}`} onClick={handleCardClick} className="block w-full h-full">
            <img
              src={imgError ? fallbackSvg : product.image_url}
              alt={product.name}
              onError={() => setImgError(true)}
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
              loading="lazy"
            />
          </Link>

          {/* Discount Tag */}
          {product.discount_pct > 0 && (
            <span className="absolute top-2.5 left-2.5 bg-rose-500 text-white text-[11px] font-bold px-2 py-0.5 rounded-full shadow-sm">
              {Math.round(product.discount_pct)}% OFF
            </span>
          )}

          {/* Exploration Badge if active */}
          {was_exploration && (
            <span className="absolute top-2.5 right-11 bg-amber-500 text-white text-[10px] font-extrabold px-1.5 py-0.5 rounded shadow-sm">
              EXPLORE
            </span>
          )}

          {/* Wishlist Button */}
          <button
            onClick={handleWishlistToggle}
            className={`absolute top-2.5 right-2.5 p-1.5 rounded-full backdrop-blur-md shadow-sm transition ${
              isFav
                ? 'bg-rose-50 text-rose-600 dark:bg-rose-950/80 dark:text-rose-400'
                : 'bg-white/80 dark:bg-slate-900/80 text-slate-400 hover:text-rose-500 hover:bg-white'
            }`}
            title={isFav ? 'Remove from wishlist' : 'Save to wishlist'}
          >
            <Heart className={`w-4 h-4 ${isFav ? 'fill-current' : ''}`} />
          </button>

          {/* Dislike / Not Interested Button on recommendation card */}
          {explanation && (
            <button
              onClick={handleDislike}
              className="absolute bottom-2.5 right-2.5 p-1.5 rounded-full bg-white/85 dark:bg-slate-900/85 text-slate-400 hover:text-rose-600 hover:bg-white shadow-sm transition"
              title="Not interested / Dislike (tunes your recommendation bandit)"
            >
              <ThumbsDown className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Explanation Chip */}
        {explanation && (
          <div className="mb-2 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-100 dark:border-indigo-900/50 text-[11px] font-medium text-indigo-700 dark:text-indigo-300">
            <Sparkles className="w-3 h-3 text-indigo-500 shrink-0" />
            <span className="truncate">{explanation}</span>
          </div>
        )}

        {/* Category & Rating */}
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
          <span className="font-semibold uppercase tracking-wider text-[10px]">
            {product.brand || product.category}
          </span>
          <div className="flex items-center gap-1 font-bold text-amber-500">
            <Star className="w-3.5 h-3.5 fill-current text-amber-400" />
            <span>{product.rating.toFixed(1)}</span>
            <span className="text-slate-400 font-normal">({product.rating_count})</span>
          </div>
        </div>

        {/* Title */}
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white line-clamp-2 mb-2 leading-snug">
          <Link to={`/products/${product.id}`} onClick={handleCardClick} className="hover:text-indigo-600 dark:hover:text-indigo-400 transition">
            {product.name}
          </Link>
        </h3>
      </div>

      {/* Pricing & Add to Cart */}
      <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between gap-2 mt-auto">
        <div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-base font-extrabold text-slate-900 dark:text-white">
              ₹{product.price.toLocaleString('en-IN')}
            </span>
            {product.mrp > product.price && (
              <span className="text-xs text-slate-400 line-through">
                ₹{product.mrp.toLocaleString('en-IN')}
              </span>
            )}
          </div>
        </div>

        <button
          onClick={handleAddToCart}
          disabled={adding || product.stock <= 0}
          className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 shadow-sm transition ${
            product.stock <= 0
              ? 'bg-slate-200 text-slate-400 cursor-not-allowed dark:bg-slate-800'
              : adding
              ? 'bg-emerald-600 text-white'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white active:scale-95'
          }`}
        >
          <ShoppingBag className="w-3.5 h-3.5" />
          <span>{product.stock <= 0 ? 'Out of Stock' : adding ? 'Added' : 'Add'}</span>
        </button>
      </div>
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Star,
  Heart,
  ShoppingBag,
  ShieldCheck,
  Truck,
  RotateCcw,
  Sparkles,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import ProductCard from '../components/ProductCard';
import { useCartWishlist } from '../context/CartWishlistContext';
import { api } from '../api/client';

export default function ProductDetailPage() {
  const { id } = useParams();
  const [product, setProduct] = useState(null);
  const [similar, setSimilar] = useState([]);
  const [quantity, setQuantity] = useState(1);
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const { addToCart, addToWishlist, removeFromWishlist, inWishlist } = useCartWishlist();

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const [prodRes, simRes] = await Promise.all([
          api.getProduct(id),
          api.getSimilarProducts(id),
        ]);
        setProduct(prodRes.data);
        setSimilar(simRes.data || []);
      } catch (e) {
        console.error('Failed to load product detail:', e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [id]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-16 animate-pulse">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
          <div className="aspect-square bg-slate-200 dark:bg-slate-800 rounded-3xl" />
          <div className="space-y-4">
            <div className="h-8 bg-slate-200 dark:bg-slate-800 rounded w-3/4" />
            <div className="h-6 bg-slate-200 dark:bg-slate-800 rounded w-1/3" />
            <div className="h-24 bg-slate-200 dark:bg-slate-800 rounded" />
          </div>
        </div>
      </div>
    );
  }

  if (!product) {
    return (
      <div className="text-center py-20">
        <h2 className="text-xl font-bold">Product not found</h2>
        <Link to="/products" className="text-indigo-600 font-semibold mt-2 inline-block">
          ← Back to Catalog
        </Link>
      </div>
    );
  }

  const isFav = inWishlist(product.id);

  const handleAddToCart = async () => {
    setAdding(true);
    await addToCart(product.id, quantity);
    setTimeout(() => setAdding(false), 800);
  };

  const handleWishlistToggle = async () => {
    if (isFav) {
      await removeFromWishlist(product.id);
    } else {
      await addToWishlist(product.id);
    }
  };

  const fallbackSvg = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 0 600 600" fill="%23f8fafc"><rect width="600" height="600" fill="%23f1f5f9"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="24" fill="%2394a3b8">${encodeURIComponent(product.category)}</text></svg>`;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-16">
      {/* Product Main Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-start">
        {/* Left: Product Image */}
        <div className="relative aspect-square w-full rounded-3xl overflow-hidden bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-800 shadow-md">
          <img
            src={product.image_url}
            alt={product.name}
            onError={(e) => {
              e.target.src = fallbackSvg;
            }}
            className="w-full h-full object-cover"
          />
          {product.discount_pct > 0 && (
            <span className="absolute top-4 left-4 bg-rose-500 text-white text-xs font-bold px-3 py-1 rounded-full shadow-md">
              {Math.round(product.discount_pct)}% OFF
            </span>
          )}
        </div>

        {/* Right: Product Details & Purchase Form */}
        <div className="space-y-6">
          <div>
            <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 mb-1">
              <span>{product.brand} • {product.category}</span>
              <span className="text-slate-400">ID: #{product.id}</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white leading-tight">
              {product.name}
            </h1>

            {/* Rating Stars */}
            <div className="flex items-center gap-2 mt-2">
              <div className="flex items-center gap-1 bg-amber-50 dark:bg-amber-950/60 border border-amber-200 dark:border-amber-800/80 px-2.5 py-1 rounded-lg text-xs font-bold text-amber-900 dark:text-amber-300">
                <Star className="w-4 h-4 fill-current text-amber-500" />
                <span>{product.rating.toFixed(1)}</span>
              </div>
              <span className="text-xs text-slate-500">
                ({product.rating_count.toLocaleString('en-IN')} verified customer ratings)
              </span>
            </div>
          </div>

          {/* Pricing */}
          <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60">
            <div className="flex items-baseline gap-3">
              <span className="text-3xl font-black text-slate-900 dark:text-white">
                ₹{product.price.toLocaleString('en-IN')}
              </span>
              {product.mrp > product.price && (
                <span className="text-base text-slate-400 line-through">
                  ₹{product.mrp.toLocaleString('en-IN')}
                </span>
              )}
              {product.discount_pct > 0 && (
                <span className="text-xs font-bold text-emerald-600 bg-emerald-50 dark:bg-emerald-950 px-2 py-0.5 rounded">
                  Save ₹{(product.mrp - product.price).toLocaleString('en-IN')}
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">Inclusive of all taxes • Cash on Delivery & UPI available</p>
          </div>

          {/* Stock Status & Actions */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-xs font-semibold">
              {product.stock > 0 ? (
                <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="w-4 h-4" /> In Stock ({product.stock} units available)
                </span>
              ) : (
                <span className="flex items-center gap-1.5 text-rose-500">
                  <AlertCircle className="w-4 h-4" /> Currently Out of Stock
                </span>
              )}
            </div>

            <div className="flex items-center gap-4">
              {/* Quantity Stepper */}
              <div className="flex items-center border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden text-sm font-semibold">
                <button
                  onClick={() => setQuantity(Math.max(1, quantity - 1))}
                  className="px-3 py-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200"
                >
                  -
                </button>
                <span className="px-4 py-2 text-center min-w-[2.5rem]">{quantity}</span>
                <button
                  onClick={() => setQuantity(Math.min(product.stock, quantity + 1))}
                  className="px-3 py-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200"
                >
                  +
                </button>
              </div>

              {/* Add to Cart Button */}
              <button
                onClick={handleAddToCart}
                disabled={adding || product.stock <= 0}
                className={`flex-1 py-3 px-6 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-lg transition ${
                  product.stock <= 0
                    ? 'bg-slate-300 text-slate-500 cursor-not-allowed'
                    : adding
                    ? 'bg-emerald-600 text-white'
                    : 'bg-indigo-600 hover:bg-indigo-500 text-white active:scale-95'
                }`}
              >
                <ShoppingBag className="w-4 h-4" />
                <span>{adding ? 'Added to Cart!' : 'Add to Cart'}</span>
              </button>

              {/* Wishlist Button */}
              <button
                onClick={handleWishlistToggle}
                className={`p-3 rounded-xl border border-slate-200 dark:border-slate-700 transition ${
                  isFav
                    ? 'bg-rose-50 text-rose-600 border-rose-300 dark:bg-rose-950/60'
                    : 'hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500'
                }`}
                title={isFav ? 'Remove from Wishlist' : 'Add to Wishlist'}
              >
                <Heart className={`w-5 h-5 ${isFav ? 'fill-current' : ''}`} />
              </button>
            </div>
          </div>

          {/* Delivery & Warranty Highlights */}
          <div className="grid grid-cols-3 gap-3 pt-4 border-t border-slate-100 dark:border-slate-800 text-center text-xs">
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40">
              <Truck className="w-4 h-4 mx-auto text-indigo-500 mb-1" />
              <span className="font-semibold block text-slate-800 dark:text-slate-200">Express Delivery</span>
              <span className="text-[10px] text-slate-400">Within 2-4 days</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40">
              <RotateCcw className="w-4 h-4 mx-auto text-emerald-500 mb-1" />
              <span className="font-semibold block text-slate-800 dark:text-slate-200">7-Day Return</span>
              <span className="text-[10px] text-slate-400">Easy refunds</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40">
              <ShieldCheck className="w-4 h-4 mx-auto text-purple-500 mb-1" />
              <span className="font-semibold block text-slate-800 dark:text-slate-200">Warranty</span>
              <span className="text-[10px] text-slate-400">Verified retailer</span>
            </div>
          </div>

          {/* Specifications Table */}
          {product.attributes && Object.keys(product.attributes).length > 0 && (
            <div className="pt-4 border-t border-slate-100 dark:border-slate-800">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                Product Specifications
              </h3>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                {Object.entries(product.attributes).map(([key, val]) => (
                  <div key={key} className="py-1 border-b border-slate-100 dark:border-slate-800/60">
                    <dt className="text-slate-400 font-medium">{key}</dt>
                    <dd className="font-semibold text-slate-800 dark:text-slate-200 truncate">{String(val)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          {/* Description */}
          <div className="pt-4 border-t border-slate-100 dark:border-slate-800">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Description</h3>
            <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              {product.description}
            </p>
          </div>
        </div>
      </div>

      {/* Similar Products Row */}
      {similar.length > 0 && (
        <section className="pt-10 border-t border-slate-200 dark:border-slate-800">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white">
              Similar in {product.category}
            </h2>
            <Link
              to={`/products?category=${encodeURIComponent(product.category)}`}
              className="text-xs font-bold text-indigo-600 hover:underline"
            >
              Explore Category →
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {similar.slice(0, 4).map((p) => (
              <ProductCard key={p.id} product={p} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}


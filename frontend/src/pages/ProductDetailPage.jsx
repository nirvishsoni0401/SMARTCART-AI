import React, { useState, useEffect, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Star,
  Heart,
  ShoppingBag,
  ShieldCheck,
  Truck,
  RotateCcw,
  CheckCircle2,
  MessageSquare,
  UserCheck,
  ZoomIn,
  X
} from 'lucide-react';
import ProductCard from '../components/ProductCard';
import RecentlyViewedRow from '../components/RecentlyViewedRow';
import ProductImage from '../components/ProductImage';
import { useCartWishlist } from '../context/CartWishlistContext';
import { api } from '../api/client';

export default function ProductDetailPage() {
  const { id } = useParams();
  const [product, setProduct] = useState(null);
  const [similar, setSimilar] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isZoomed, setIsZoomed] = useState(false);
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
        setSelectedImage(prodRes.data?.image_url || null);
        setSimilar(simRes.data || []);
      } catch (e) {
        console.error('Failed to load product detail:', e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [id]);

  // Close zoom modal on Escape
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isZoomed) {
        setIsZoomed(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isZoomed]);

  // Gallery candidates list - unconditionally defined above early returns
  const galleryImages = useMemo(() => {
    if (!product) return [];
    const list = [];
    const seen = new Set();
    const add = (u) => {
      if (u && typeof u === 'string' && u.trim() && !seen.has(u.trim())) {
        seen.add(u.trim());
        list.push(u.trim());
      }
    };
    if (product.attributes?.local_image_url) add(product.attributes.local_image_url);
    if (product.image_url) add(product.image_url);
    if (Array.isArray(product.attributes?.local_gallery)) product.attributes.local_gallery.forEach(add);
    if (Array.isArray(product.attributes?.gallery)) product.attributes.gallery.forEach(add);
    if (product.attributes?.cdn_image_url) add(product.attributes.cdn_image_url);
    return list.length > 0 ? list : (product?.image_url ? [product.image_url] : []);
  }, [product]);

  const isFav = product ? inWishlist(product.id) : false;

  const handleAddToCart = async () => {
    if (!product) return;
    setAdding(true);
    await addToCart(product.id, quantity);
    setTimeout(() => setAdding(false), 800);
  };

  const handleWishlistToggle = async () => {
    if (!product) return;
    if (isFav) {
      await removeFromWishlist(product.id);
    } else {
      await addToWishlist(product.id);
    }
  };

  const handleThumbnailKeyDown = (e, idx) => {
    if (!galleryImages || galleryImages.length === 0) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      const nextIdx = (idx + 1) % galleryImages.length;
      setSelectedImage(galleryImages[nextIdx]);
      document.getElementById(`gallery-thumb-${nextIdx}`)?.focus();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      const prevIdx = (idx - 1 + galleryImages.length) % galleryImages.length;
      setSelectedImage(galleryImages[prevIdx]);
      document.getElementById(`gallery-thumb-${prevIdx}`)?.focus();
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setSelectedImage(galleryImages[idx]);
    }
  };

  // Early returns strictly placed AFTER all hooks
  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-16 animate-pulse" data-testid="product-loading">
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
      <div className="text-center py-20" data-testid="product-not-found">
        <h2 className="text-xl font-bold">Product not found or inactive</h2>
        <Link to="/products" className="text-indigo-600 font-semibold mt-2 inline-block">
          ← Back to Catalog
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      {/* Click-to-Zoom Lightbox Modal */}
      {isZoomed && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Enlarged product image preview"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md"
          onClick={() => setIsZoomed(false)}
        >
          <div
            className="relative max-w-3xl w-full flex flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setIsZoomed(false)}
              className="absolute -top-12 right-0 p-2 text-white/80 hover:text-white bg-white/10 hover:bg-white/20 rounded-full transition focus:outline-none focus:ring-2 focus:ring-white"
              aria-label="Close zoomed image"
            >
              <X className="w-6 h-6" />
            </button>
            <div className="w-full aspect-square bg-white dark:bg-slate-900 rounded-3xl p-6 sm:p-10 shadow-2xl border border-slate-700/50 flex items-center justify-center overflow-hidden">
              <ProductImage
                product={product}
                src={selectedImage}
                aspectRatio="aspect-square"
                hoverZoom={false}
                eager={true}
              />
            </div>
            <p className="text-xs text-slate-300 mt-3 text-center">
              Press Escape or click anywhere outside to close
            </p>
          </div>
        </div>
      )}

      {/* Breadcrumb Navigation */}
      <nav className="flex items-center gap-2 text-xs text-slate-500">
        <Link to="/" className="hover:text-indigo-600">Home</Link>
        <span>/</span>
        <Link to="/products" className="hover:text-indigo-600">Products</Link>
        <span>/</span>
        <Link to={`/products?category=${encodeURIComponent(product.category)}`} className="hover:text-indigo-600 font-medium">
          {product.category}
        </Link>
        <span>/</span>
        <span className="text-slate-900 dark:text-slate-200 truncate max-w-xs">{product.name}</span>
      </nav>

      {/* Main Product Container */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-14">
        {/* Left: Gallery & Main Image */}
        <div className="space-y-4">
          <div className="aspect-square rounded-3xl bg-white dark:bg-slate-900 p-4 sm:p-8 flex items-center justify-center border border-slate-200 dark:border-slate-800 shadow-sm relative overflow-hidden group">
            <button
              type="button"
              onClick={() => setIsZoomed(true)}
              className="w-full h-full block focus:outline-none focus-visible:ring-4 focus-visible:ring-indigo-500 rounded-2xl cursor-zoom-in relative"
              aria-label="Click to enlarge product photo"
            >
              <ProductImage
                product={product}
                src={selectedImage}
                aspectRatio="aspect-square"
                eager={true}
                hoverZoom={true}
              />
              <span className="absolute bottom-3 right-3 bg-black/60 text-white text-[11px] font-semibold px-2.5 py-1 rounded-lg backdrop-blur-sm opacity-0 group-hover:opacity-100 transition flex items-center gap-1.5 shadow-sm">
                <ZoomIn className="w-3.5 h-3.5" />
                <span>Zoom</span>
              </span>
            </button>

            {product.discount_pct > 0 && (
              <span className="absolute top-4 left-4 bg-rose-500 text-white text-xs font-black px-2.5 py-1 rounded-full shadow z-10">
                {Math.round(product.discount_pct)}% OFF
              </span>
            )}
            {product.attributes?.is_unbranded && (
              <span className="absolute top-4 right-4 bg-slate-100 dark:bg-slate-800 text-slate-500 text-[11px] font-semibold px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-700 z-10">
                Direct Catalog Item
              </span>
            )}
          </div>

          {/* Multiple Image Thumbnails */}
          {galleryImages.length > 1 && (
            <div
              role="tablist"
              aria-label="Product image thumbnails"
              className="flex items-center gap-3 overflow-x-auto pb-2"
            >
              {galleryImages.map((imgUrl, idx) => (
                <button
                  key={idx}
                  id={`gallery-thumb-${idx}`}
                  role="tab"
                  aria-selected={selectedImage === imgUrl}
                  aria-label={`Select product image ${idx + 1} of ${galleryImages.length}`}
                  onClick={() => setSelectedImage(imgUrl)}
                  onKeyDown={(e) => handleThumbnailKeyDown(e, idx)}
                  tabIndex={0}
                  className={`w-16 h-16 rounded-xl bg-white dark:bg-slate-900 p-1 border-2 transition flex-shrink-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                    selectedImage === imgUrl
                      ? 'border-indigo-600 shadow-md ring-2 ring-indigo-200 dark:ring-indigo-900/60'
                      : 'border-slate-200 dark:border-slate-800 hover:border-slate-400'
                  }`}
                >
                  <ProductImage
                    product={product}
                    src={imgUrl}
                    aspectRatio="aspect-square"
                    hoverZoom={false}
                  />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right: Product Details & Purchase Actions */}
        <div className="space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-extrabold uppercase tracking-wider text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950/60 px-2.5 py-0.5 rounded-md">
                {product.brand}
              </span>
              <span className="text-xs text-slate-400 font-medium">• {product.category}</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white break-words">
              {product.name}
            </h1>
          </div>

          {/* Rating and Social Proof */}
          <div className="flex items-center gap-4 text-xs">
            <div className="inline-flex items-center gap-1 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 px-2.5 py-1 rounded-lg text-amber-700 dark:text-amber-400 font-bold">
              <Star className="w-3.5 h-3.5 fill-amber-400 text-amber-400" />
              <span>{product.rating.toFixed(1)}</span>
            </div>
            <span className="text-slate-500 font-medium">
              {product.rating_count} verified reviews
            </span>
            <span className="text-slate-300 dark:text-slate-700">|</span>
            <div className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{product.stock > 0 ? `In Stock (${product.stock} units)` : 'Out of Stock'}</span>
            </div>
          </div>

          {/* Pricing Block */}
          <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-900/60 border border-slate-100 dark:border-slate-800 flex items-baseline gap-3">
            <span className="text-3xl font-black text-slate-900 dark:text-white">
              ₹{product.price.toLocaleString('en-IN')}
            </span>
            {product.mrp > product.price && (
              <span className="text-base text-slate-400 line-through">
                ₹{product.mrp.toLocaleString('en-IN')}
              </span>
            )}
            {product.discount_pct > 0 && (
              <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/50 px-2 py-0.5 rounded-full">
                Save ₹{(product.mrp - product.price).toLocaleString('en-IN')} ({Math.round(product.discount_pct)}%)
              </span>
            )}
          </div>

          {/* Quantity and Actions */}
          <div className="space-y-4 pt-2">
            <div className="flex items-center gap-4">
              <div className="flex items-center border border-slate-200 dark:border-slate-700 rounded-xl bg-white dark:bg-slate-900 overflow-hidden">
                <button
                  onClick={() => setQuantity((q) => Math.max(1, q - 1))}
                  className="px-3 py-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition text-sm font-bold"
                >
                  -
                </button>
                <span className="px-4 py-2 text-sm font-bold text-slate-800 dark:text-slate-200">
                  {quantity}
                </span>
                <button
                  onClick={() => setQuantity((q) => Math.min(product.stock, q + 1))}
                  className="px-3 py-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition text-sm font-bold"
                >
                  +
                </button>
              </div>

              {/* Add to Cart Button */}
              <button
                onClick={handleAddToCart}
                disabled={product.stock <= 0}
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
              <span className="font-semibold block text-slate-800 dark:text-slate-200">Delivery</span>
              <span className="text-[10px] text-slate-400">{product.attributes?.shippingInformation || 'Ships in 2-4 days'}</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40">
              <RotateCcw className="w-4 h-4 mx-auto text-emerald-500 mb-1" />
              <span className="font-semibold block text-slate-800 dark:text-slate-200">Return Policy</span>
              <span className="text-[10px] text-slate-400">{product.attributes?.returnPolicy || '7-Day Return'}</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/40">
              <ShieldCheck className="w-4 h-4 mx-auto text-purple-500 mb-1" />
              <span className="font-semibold block text-slate-800 dark:text-slate-200">Warranty</span>
              <span className="text-[10px] text-slate-400">{product.attributes?.warrantyInformation || 'Standard Warranty'}</span>
            </div>
          </div>

          {/* Description */}
          <div className="pt-4 border-t border-slate-100 dark:border-slate-800">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Description</h3>
            <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
              {product.description}
            </p>
          </div>

          {/* Specifications Table */}
          {product.attributes && Object.keys(product.attributes).length > 0 && (
            <div className="pt-4 border-t border-slate-100 dark:border-slate-800">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                Product Details & Attributes
              </h3>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                {product.attributes.sku && (
                  <div className="py-1 border-b border-slate-100 dark:border-slate-800/60">
                    <dt className="text-slate-400 font-medium">SKU</dt>
                    <dd className="font-semibold text-slate-800 dark:text-slate-200">{product.attributes.sku}</dd>
                  </div>
                )}
                {product.attributes.weight && (
                  <div className="py-1 border-b border-slate-100 dark:border-slate-800/60">
                    <dt className="text-slate-400 font-medium">Weight</dt>
                    <dd className="font-semibold text-slate-800 dark:text-slate-200">{product.attributes.weight} kg</dd>
                  </div>
                )}
                {product.attributes.availabilityStatus && (
                  <div className="py-1 border-b border-slate-100 dark:border-slate-800/60">
                    <dt className="text-slate-400 font-medium">Availability</dt>
                    <dd className="font-semibold text-slate-800 dark:text-slate-200">{product.attributes.availabilityStatus}</dd>
                  </div>
                )}
                <div className="py-1 border-b border-slate-100 dark:border-slate-800/60">
                  <dt className="text-slate-400 font-medium">Catalog Source</dt>
                  <dd className="font-semibold text-slate-800 dark:text-slate-200 capitalize">{product.source || 'Standard'}</dd>
                </div>
              </dl>
            </div>
          )}
        </div>
      </div>

      {/* Verified Customer Reviews Section (Phase 8 B5) */}
      <section className="pt-10 border-t border-slate-200 dark:border-slate-800 space-y-6">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-indigo-600" />
          <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white">
            Verified Customer Reviews
          </h2>
          <span className="ml-2 text-xs bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded-full text-slate-600 dark:text-slate-400 font-bold">
            {product.reviews?.length || 0} reviews
          </span>
        </div>

        {product.reviews && product.reviews.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {product.reviews.map((rev) => (
              <div
                key={rev.id}
                className="p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm space-y-3"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <UserCheck className="w-4 h-4 text-emerald-500" />
                    <span className="font-bold text-xs text-slate-900 dark:text-white truncate">
                      {rev.reviewer_name}
                    </span>
                  </div>
                  <div className="flex items-center text-amber-400">
                    {[...Array(5)].map((_, i) => (
                      <Star
                        key={i}
                        className={`w-3 h-3 ${
                          i < Math.round(rev.rating) ? 'fill-current' : 'text-slate-200 dark:text-slate-700'
                        }`}
                      />
                    ))}
                  </div>
                </div>

                <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed italic">
                  "{rev.comment}"
                </p>

                <div className="flex items-center justify-between text-[10px] text-slate-400 pt-2 border-t border-slate-100 dark:border-slate-800">
                  <span>Verified Buyer</span>
                  {rev.date && <span>{new Date(rev.date).toLocaleDateString()}</span>}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-400 italic">No customer reviews yet for this product.</p>
        )}
      </section>

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

      {/* Recently Viewed Products Row (User History, Non-Bandit) */}
      <RecentlyViewedRow title="Recently Viewed" />
    </div>
  );
}

import React, { useState, useEffect, useMemo } from 'react';
import {
  BookOpen,
  Laptop,
  Shirt,
  Home as HomeIcon,
  Sparkles,
  Dumbbell,
  Package,
} from 'lucide-react';

const rawBase = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) || 'http://localhost:8000';
export const API_BASE_URL = rawBase.replace(/\/+$/, '');

/**
 * Resolves relative static paths (/static/...) to backend server URL.
 * Preserves external CDN URLs, data URIs, and blob URIs.
 */
export function resolveImageUrl(url) {
  if (!url || typeof url !== 'string') return '';
  const trimmed = url.trim();
  if (!trimmed) return '';
  if (
    trimmed.startsWith('http://') ||
    trimmed.startsWith('https://') ||
    trimmed.startsWith('data:') ||
    trimmed.startsWith('blob:')
  ) {
    return trimmed;
  }
  if (trimmed.startsWith('/static') || trimmed.startsWith('static/')) {
    const cleanPath = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
    return `${API_BASE_URL}${cleanPath}`;
  }
  if (trimmed.startsWith('/')) {
    return `${API_BASE_URL}${trimmed}`;
  }
  return trimmed;
}

/**
 * Department to Lucide icon mapping
 */
const DEPARTMENT_ICONS = {
  Books: BookOpen,
  Electronics: Laptop,
  Fashion: Shirt,
  'Home & Kitchen': HomeIcon,
  Beauty: Sparkles,
  Sports: Dumbbell,
};

/**
 * Returns 1-2 uppercase initials from a product name.
 */
function getInitials(name) {
  if (!name) return 'SC';
  const clean = name.replace(/[^a-zA-Z0-9\s]/g, ' ').trim();
  const words = clean.split(/\s+/).filter(Boolean);
  if (words.length === 0) return 'SC';
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

/**
 * Deterministic fallback placeholder rendered when all image URLs fail.
 */
export function ProductImagePlaceholder({ product, className = '' }) {
  const category = product?.category || 'General';
  const name = product?.name || 'SmartCart Product';
  const brand = product?.brand || '';
  const Icon = DEPARTMENT_ICONS[category] || Package;
  const initials = getInitials(name);

  return (
    <div
      role="img"
      aria-label={`${name} (${category})`}
      className={`w-full h-full flex flex-col items-center justify-center p-4 text-center select-none bg-gradient-to-br from-slate-100 via-slate-200 to-indigo-50 dark:from-slate-800 dark:via-slate-850 dark:to-slate-900 border border-slate-200/80 dark:border-slate-700/80 transition-colors ${className}`}
    >
      <div className="w-12 h-12 sm:w-14 sm:h-14 rounded-2xl bg-white/90 dark:bg-slate-700/90 shadow-sm border border-slate-200 dark:border-slate-600 flex items-center justify-center text-indigo-600 dark:text-indigo-400 mb-2">
        <Icon className="w-6 h-6 sm:w-7 sm:h-7" aria-hidden="true" />
      </div>
      <div className="text-sm sm:text-base font-extrabold tracking-wider text-slate-800 dark:text-slate-100">
        {initials}
      </div>
      <div className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mt-0.5 truncate max-w-full px-2">
        {category}
      </div>
      {brand && (
        <div className="text-[9px] text-slate-400 dark:text-slate-500 truncate max-w-full mt-0.5">
          {brand}
        </div>
      )}
    </div>
  );
}

/**
 * Reusable ProductImage component with:
 * - Fixed aspect ratio container (zero layout shift)
 * - Shimmer skeleton while loading
 * - Lazy / async decoding by default (eager for fold)
 * - Robust onError fallback chain (primary -> local -> gallery -> placeholder)
 * - Object-fit: cover for books, contain for product photos
 * - Subtle hover zoom with reduced-motion support
 */
export default function ProductImage({
  product,
  src = null,
  alt = null,
  aspectRatio = 'aspect-square',
  eager = false,
  hoverZoom = true,
  fit = null,
  className = '',
  imgClassName = '',
  onClick = null,
}) {
  const isBook = product?.category === 'Books';
  // Books use cover; other product photos use contain to avoid cropping
  const objectFitClass = fit
    ? fit === 'cover'
      ? 'object-cover'
      : 'object-contain'
    : isBook
    ? 'object-cover'
    : 'object-contain';

  // Build candidate URL list
  const candidates = useMemo(() => {
    const list = [];
    const seen = new Set();

    const addUrl = (u) => {
      if (u && typeof u === 'string' && u.trim() && !seen.has(u.trim())) {
        seen.add(u.trim());
        list.push(resolveImageUrl(u.trim()));
      }
    };

    // 1. Explicit src override if provided
    if (src) addUrl(src);

    // 2. Local cached image if recorded in attributes
    if (product?.attributes?.local_image_url) {
      addUrl(product.attributes.local_image_url);
    }

    // 3. Primary image_url
    if (product?.image_url) {
      addUrl(product.image_url);
    }

    // 4. Local gallery images
    if (Array.isArray(product?.attributes?.local_gallery)) {
      product.attributes.local_gallery.forEach(addUrl);
    }

    // 5. Remote gallery images
    if (Array.isArray(product?.attributes?.gallery)) {
      product.attributes.gallery.forEach(addUrl);
    }

    // 6. CDN image url fallback
    if (product?.attributes?.cdn_image_url) {
      addUrl(product.attributes.cdn_image_url);
    }

    return list;
  }, [src, product]);

  const [candidateIndex, setCandidateIndex] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [allFailed, setAllFailed] = useState(false);

  // Reset when candidates change (e.g. selectedImage changed)
  useEffect(() => {
    setCandidateIndex(0);
    setIsLoaded(false);
    setAllFailed(candidates.length === 0);
  }, [candidates]);

  const currentUrl = candidates[candidateIndex] || null;

  const handleImageError = () => {
    if (candidateIndex + 1 < candidates.length) {
      setCandidateIndex((prev) => prev + 1);
      setIsLoaded(false);
    } else {
      setAllFailed(true);
      setIsLoaded(true);
    }
  };

  const handleImageLoad = () => {
    setIsLoaded(true);
  };

  const meaningfulAlt =
    alt ||
    (product?.name
      ? `${product.name}${product.brand ? ` by ${product.brand}` : ''} - ${product.category || 'Product'}`
      : 'Product image');

  return (
    <div
      onClick={onClick}
      className={`relative w-full ${aspectRatio} overflow-hidden rounded-xl bg-slate-100 dark:bg-slate-800 ${className}`}
    >
      {/* Loading Shimmer Skeleton */}
      {!isLoaded && !allFailed && (
        <div
          aria-hidden="true"
          className="absolute inset-0 z-0 bg-slate-200 dark:bg-slate-800 animate-pulse motion-reduce:animate-none"
        >
          <div className="w-full h-full bg-gradient-to-r from-transparent via-white/20 dark:via-white/5 to-transparent -translate-x-full animate-[shimmer_1.5s_infinite] motion-reduce:animate-none" />
        </div>
      )}

      {/* Fallback Placeholder when all candidates fail */}
      {allFailed ? (
        <ProductImagePlaceholder product={product} />
      ) : (
        <img
          src={currentUrl}
          alt={meaningfulAlt}
          loading={eager ? 'eager' : 'lazy'}
          decoding="async"
          referrerPolicy="no-referrer"
          onLoad={handleImageLoad}
          onError={handleImageError}
          className={`w-full h-full ${objectFitClass} transition-all duration-300 motion-reduce:transition-none ${
            isLoaded ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
          } ${
            hoverZoom ? 'group-hover:scale-105 motion-reduce:group-hover:scale-100' : ''
          } ${imgClassName}`}
        />
      )}
    </div>
  );
}

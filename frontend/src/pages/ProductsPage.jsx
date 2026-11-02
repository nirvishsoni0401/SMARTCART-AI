import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Filter, X, SlidersHorizontal, ArrowUpDown } from 'lucide-react';
import ProductCard from '../components/ProductCard';
import { api } from '../api/client';

export default function ProductsPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Query state
  const categoryParam = searchParams.get('category') || '';
  const searchParam = searchParams.get('q') || '';
  const brandParam = searchParams.get('brand') || '';
  const minPriceParam = searchParams.get('min_price') || '';
  const maxPriceParam = searchParams.get('max_price') || '';
  const minRatingParam = searchParams.get('min_rating') || '';
  const minDiscountParam = searchParams.get('min_discount') || '';
  const sortParam = searchParams.get('sort_by') || 'popularity';
  const pageParam = parseInt(searchParams.get('page') || '1', 10);

  const [products, setProducts] = useState([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [categories, setCategories] = useState([]);
  const [brands, setBrands] = useState([]);
  const [loading, setLoading] = useState(true);
  const [mobileFilterOpen, setMobileFilterOpen] = useState(false);

  // Fetch categories & brands
  useEffect(() => {
    api.getCategories().then((res) => setCategories(res.data || []));
    api.getBrands(categoryParam).then((res) => setBrands(res.data || []));
  }, [categoryParam]);

  // Fetch products
  useEffect(() => {
    setLoading(true);
    api
      .getProducts({
        q: searchParam || undefined,
        category: categoryParam || undefined,
        brand: brandParam || undefined,
        min_price: minPriceParam ? parseFloat(minPriceParam) : undefined,
        max_price: maxPriceParam ? parseFloat(maxPriceParam) : undefined,
        min_rating: minRatingParam ? parseFloat(minRatingParam) : undefined,
        min_discount: minDiscountParam ? parseFloat(minDiscountParam) : undefined,
        sort_by: sortParam,
        page: pageParam,
        page_size: 20,
      })
      .then((res) => {
        setProducts(res.data.products || []);
        setTotal(res.data.total || 0);
        setTotalPages(res.data.total_pages || 1);
      })
      .catch((e) => console.error('Error fetching products:', e))
      .finally(() => setLoading(false));
  }, [
    categoryParam,
    searchParam,
    brandParam,
    minPriceParam,
    maxPriceParam,
    minRatingParam,
    minDiscountParam,
    sortParam,
    pageParam,
  ]);

  const updateParam = (key, value) => {
    const next = new URLSearchParams(searchParams);
    if (value === null || value === '' || value === undefined) {
      next.delete(key);
    } else {
      next.set(key, value);
    }
    next.set('page', '1'); // reset page on filter change
    setSearchParams(next);
  };

  const clearAllFilters = () => {
    setSearchParams(new URLSearchParams());
  };

  const hasActiveFilters = Boolean(
    categoryParam || searchParam || brandParam || minPriceParam || maxPriceParam || minRatingParam || minDiscountParam
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-200 dark:border-slate-800 mb-6">
        <div>
          <h1 className="text-2xl font-black text-slate-900 dark:text-white">
            {categoryParam || (searchParam ? `Search: "${searchParam}"` : 'All Products')}
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Showing {products.length} of {total} products in Indian Rupees (₹)
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Mobile Filter Toggle */}
          <button
            onClick={() => setMobileFilterOpen(!mobileFilterOpen)}
            className="md:hidden flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-100 dark:bg-slate-800 text-xs font-semibold"
          >
            <SlidersHorizontal className="w-4 h-4" />
            <span>Filters</span>
          </button>

          {/* Sort By Dropdown */}
          <div className="flex items-center gap-2 text-xs">
            <ArrowUpDown className="w-4 h-4 text-slate-400" />
            <select
              value={sortParam}
              onChange={(e) => updateParam('sort_by', e.target.value)}
              className="px-3 py-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl text-xs font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="popularity">Most Popular</option>
              <option value="price_asc">Price: Low to High</option>
              <option value="price_desc">Price: High to Low</option>
              <option value="rating_desc">Highest Rated</option>
              <option value="newest">Newest Arrivals</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Layout: Sidebar + Grid */}
      <div className="flex gap-8">
        {/* Filter Sidebar (Desktop & Mobile Drawer) */}
        <aside
          className={`w-64 shrink-0 space-y-6 ${
            mobileFilterOpen
              ? 'fixed inset-0 z-50 bg-white dark:bg-slate-900 p-6 overflow-y-auto block'
              : 'hidden md:block'
          }`}
        >
          {mobileFilterOpen && (
            <div className="flex items-center justify-between pb-4 border-b border-slate-200 dark:border-slate-800 mb-4 md:hidden">
              <span className="font-bold text-sm">Filters</span>
              <button onClick={() => setMobileFilterOpen(false)}>
                <X className="w-5 h-5 text-slate-500" />
              </button>
            </div>
          )}

          {hasActiveFilters && (
            <button
              onClick={clearAllFilters}
              className="w-full py-1.5 px-3 rounded-lg text-xs font-semibold text-rose-600 bg-rose-50 dark:bg-rose-950/50 hover:bg-rose-100 transition"
            >
              Clear All Filters
            </button>
          )}

          {/* Category Filter */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Category</h4>
            <div className="space-y-1">
              <button
                onClick={() => updateParam('category', '')}
                className={`w-full text-left px-2.5 py-1.5 rounded-lg text-xs font-medium transition ${
                  !categoryParam
                    ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300 font-bold'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                All Categories
              </button>
              {categories.map((c) => (
                <button
                  key={c.name}
                  onClick={() => updateParam('category', c.name)}
                  className={`w-full text-left px-2.5 py-1.5 rounded-lg text-xs font-medium flex items-center justify-between transition ${
                    categoryParam.toLowerCase() === c.name.toLowerCase()
                      ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300 font-bold'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  <span className="truncate">{c.name}</span>
                  <span className="text-[10px] text-slate-400">({c.count})</span>
                </button>
              ))}
            </div>
          </div>

          {/* Brand Filter */}
          {brands.length > 0 && (
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Brand</h4>
              <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
                <button
                  onClick={() => updateParam('brand', '')}
                  className={`w-full text-left px-2.5 py-1 rounded text-xs transition ${
                    !brandParam ? 'font-bold text-indigo-600' : 'text-slate-600 dark:text-slate-400'
                  }`}
                >
                  All Brands
                </button>
                {brands.map((b) => (
                  <button
                    key={b}
                    onClick={() => updateParam('brand', b)}
                    className={`w-full text-left px-2.5 py-1 rounded text-xs truncate transition ${
                      brandParam.toLowerCase() === b.toLowerCase()
                        ? 'font-bold text-indigo-600'
                        : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500'
                    }`}
                  >
                    {b}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Price Range */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Budget Cap (₹)</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <input
                type="number"
                placeholder="Min ₹"
                value={minPriceParam}
                onChange={(e) => updateParam('min_price', e.target.value)}
                className="px-2.5 py-1.5 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-xs"
              />
              <input
                type="number"
                placeholder="Max ₹"
                value={maxPriceParam}
                onChange={(e) => updateParam('max_price', e.target.value)}
                className="px-2.5 py-1.5 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-xs"
              />
            </div>
          </div>

          {/* Minimum Rating */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Min Rating</h4>
            <div className="space-y-1 text-xs">
              {[4.5, 4.0, 3.5].map((stars) => (
                <button
                  key={stars}
                  onClick={() => updateParam('min_rating', minRatingParam === String(stars) ? '' : String(stars))}
                  className={`w-full text-left px-2.5 py-1 rounded-lg transition ${
                    minRatingParam === String(stars)
                      ? 'bg-amber-50 text-amber-900 font-bold dark:bg-amber-950/60'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  ★ {stars} & above
                </button>
              ))}
            </div>
          </div>

          {/* Minimum Discount */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Min Discount</h4>
            <div className="space-y-1 text-xs">
              {[20, 30, 50].map((d) => (
                <button
                  key={d}
                  onClick={() => updateParam('min_discount', minDiscountParam === String(d) ? '' : String(d))}
                  className={`w-full text-left px-2.5 py-1 rounded-lg transition ${
                    minDiscountParam === String(d)
                      ? 'bg-rose-50 text-rose-900 font-bold dark:bg-rose-950/60'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  {d}% or more off
                </button>
              ))}
            </div>
          </div>

          {mobileFilterOpen && (
            <button
              onClick={() => setMobileFilterOpen(false)}
              className="w-full py-2 bg-indigo-600 text-white rounded-xl text-xs font-bold mt-4"
            >
              Apply Filters
            </button>
          )}
        </aside>

        {/* Product Grid */}
        <main className="flex-1">
          {loading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div key={i} className="h-80 rounded-2xl bg-slate-100 dark:bg-slate-800 animate-pulse border border-slate-200" />
              ))}
            </div>
          ) : products.length === 0 ? (
            <div className="text-center py-20 bg-slate-50 dark:bg-slate-800/40 rounded-3xl border border-dashed border-slate-200 dark:border-slate-700">
              <p className="text-base font-semibold text-slate-700 dark:text-slate-300">
                No products found matching your active filters.
              </p>
              <p className="text-xs text-slate-400 mt-1">Try relaxing price, rating, or search filters.</p>
              <button
                onClick={clearAllFilters}
                className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-xl text-xs font-semibold"
              >
                Reset All Filters
              </button>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {products.map((p) => (
                  <ProductCard key={p.id} product={p} />
                ))}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="mt-10 flex items-center justify-center gap-2">
                  <button
                    disabled={pageParam <= 1}
                    onClick={() => updateParam('page', String(pageParam - 1))}
                    className="px-3.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs font-medium disabled:opacity-40"
                  >
                    Previous
                  </button>
                  <span className="text-xs text-slate-500 font-semibold px-2">
                    Page {pageParam} of {totalPages}
                  </span>
                  <button
                    disabled={pageParam >= totalPages}
                    onClick={() => updateParam('page', String(pageParam + 1))}
                    className="px-3.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-xs font-medium disabled:opacity-40"
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}


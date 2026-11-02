import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  ShoppingBag,
  Heart,
  Search,
  Sparkles,
  User as UserIcon,
  LogOut,
  LayoutDashboard,
  Menu,
  X,
  PackageCheck
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useCartWishlist } from '../context/CartWishlistContext';

export default function Navbar({ onOpenAuthModal }) {
  const { user, logout, isAdmin } = useAuth();
  const { cart, wishlist } = useCartWishlist();
  const [searchQuery, setSearchQuery] = useState('');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/products?q=${encodeURIComponent(searchQuery.trim())}`);
      setMobileMenuOpen(false);
    }
  };

  const categories = [
    'Electronics',
    'Fashion',
    'Home & Kitchen',
    'Beauty',
    'Sports',
    'Books',
  ];

  return (
    <header className="sticky top-0 z-30 bg-white/95 dark:bg-slate-900/95 backdrop-blur border-b border-slate-200 dark:border-slate-800">
      {/* Top Banner */}
      <div className="bg-indigo-600 text-white text-xs py-1.5 px-4 text-center font-medium">
        <span>🇮🇳 Adaptive AI Multi-Armed Bandit E-Commerce • Free Express Shipping Across India over ₹999</span>
      </div>

      {/* Main Navbar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Brand Logo */}
        <Link to="/" className="flex items-center gap-2 shrink-0">
          <div className="p-2 bg-indigo-600 text-white rounded-xl shadow-md">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xl font-black tracking-tight text-slate-900 dark:text-white">
              SmartCart<span className="text-indigo-600">.AI</span>
            </span>
          </div>
        </Link>

        {/* Global Search Bar */}
        <form onSubmit={handleSearchSubmit} className="hidden md:flex flex-1 max-w-md relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search 600+ products, brands (e.g. boAt, Peter England)..."
            className="w-full pl-10 pr-4 py-2 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white dark:focus:bg-slate-900 transition"
          />
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
        </form>

        {/* Action Controls */}
        <div className="flex items-center gap-2 sm:gap-4">
          {/* Admin Dashboard shortcut */}
          {isAdmin && (
            <Link
              to="/admin"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-purple-50 text-purple-700 dark:bg-purple-950 dark:text-purple-300 text-xs font-bold border border-purple-200 dark:border-purple-800 hover:bg-purple-100 transition"
            >
              <LayoutDashboard className="w-4 h-4" />
              <span className="hidden sm:inline">Admin</span>
            </Link>
          )}

          {/* Orders History link */}
          {user && (
            <Link
              to="/orders"
              className="p-2 text-slate-600 dark:text-slate-300 hover:text-indigo-600 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              title="My Orders & Returns"
            >
              <PackageCheck className="w-5 h-5" />
            </Link>
          )}

          {/* Wishlist Link */}
          <Link
            to="/wishlist"
            className="p-2 text-slate-600 dark:text-slate-300 hover:text-rose-500 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 relative transition"
            title="Wishlist"
          >
            <Heart className="w-5 h-5" />
            {wishlist.total_items > 0 && (
              <span className="absolute top-1 right-1 bg-rose-500 text-white text-[10px] font-bold w-4 h-4 rounded-full flex items-center justify-center">
                {wishlist.total_items}
              </span>
            )}
          </Link>

          {/* Cart Link */}
          <Link
            to="/cart"
            className="p-2 text-slate-600 dark:text-slate-300 hover:text-indigo-600 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 relative transition"
            title="Cart"
          >
            <ShoppingBag className="w-5 h-5" />
            {cart.total_items > 0 && (
              <span className="absolute top-1 right-1 bg-indigo-600 text-white text-[10px] font-bold w-4 h-4 rounded-full flex items-center justify-center">
                {cart.total_items}
              </span>
            )}
          </Link>

          {/* User Auth Menu */}
          {user ? (
            <div className="flex items-center gap-2">
              <span className="hidden sm:inline text-xs font-semibold text-slate-700 dark:text-slate-300">
                Hi, {user.name.split(' ')[0]}
              </span>
              <button
                onClick={logout}
                className="p-2 text-slate-400 hover:text-rose-500 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                title="Logout"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={onOpenAuthModal}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 dark:bg-white dark:hover:bg-slate-100 text-white dark:text-slate-900 text-xs font-bold transition shadow-sm"
            >
              <UserIcon className="w-3.5 h-3.5" />
              <span>Login</span>
            </button>
          )}

          {/* Mobile menu toggle */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 text-slate-600 dark:text-slate-300"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Category Navigation Bar (Desktop) */}
      <nav className="hidden md:flex border-t border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50 px-4 sm:px-8 py-2 justify-center gap-8 text-xs font-semibold text-slate-600 dark:text-slate-400">
        <Link to="/products" className="hover:text-indigo-600 dark:hover:text-white transition">
          All Products (600+)
        </Link>
        {categories.map((cat) => (
          <Link
            key={cat}
            to={`/products?category=${encodeURIComponent(cat)}`}
            className="hover:text-indigo-600 dark:hover:text-white transition"
          >
            {cat}
          </Link>
        ))}
      </nav>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-slate-200 dark:border-slate-800 p-4 space-y-3 bg-white dark:bg-slate-900">
          <form onSubmit={handleSearchSubmit} className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search products..."
              className="w-full pl-9 pr-4 py-2 bg-slate-100 dark:bg-slate-800 border border-slate-200 rounded-xl text-xs"
            />
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          </form>

          <div className="grid grid-cols-2 gap-2 pt-2">
            <Link
              to="/products"
              onClick={() => setMobileMenuOpen(false)}
              className="px-3 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs font-medium"
            >
              All Products
            </Link>
            {categories.map((cat) => (
              <Link
                key={cat}
                to={`/products?category=${encodeURIComponent(cat)}`}
                onClick={() => setMobileMenuOpen(false)}
                className="px-3 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-xs font-medium"
              >
                {cat}
              </Link>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}


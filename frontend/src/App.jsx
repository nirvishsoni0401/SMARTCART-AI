import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ToastProvider } from './context/ToastContext';
import { AuthProvider } from './context/AuthContext';
import { CartWishlistProvider } from './context/CartWishlistContext';

import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ConsentBanner from './components/ConsentBanner';
import AuthModal from './components/AuthModal';

import Home from './pages/Home';
import ProductsPage from './pages/ProductsPage';
import ProductDetailPage from './pages/ProductDetailPage';
import CartPage from './pages/CartPage';
import WishlistPage from './pages/WishlistPage';
import OrderHistoryPage from './pages/OrderHistoryPage';
import AdminDashboard from './pages/AdminDashboard';

export default function App() {
  const [authModalOpen, setAuthModalOpen] = useState(false);

  return (
    <BrowserRouter>
      <ToastProvider>
        <AuthProvider>
          <CartWishlistProvider>
            <div className="min-h-screen flex flex-col bg-slate-50/50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 antialiased">
              <Navbar onOpenAuthModal={() => setAuthModalOpen(true)} />
              
              <main className="flex-1">
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/products" element={<ProductsPage />} />
                  <Route path="/products/:id" element={<ProductDetailPage />} />
                  <Route path="/cart" element={<CartPage />} />
                  <Route path="/wishlist" element={<WishlistPage />} />
                  <Route path="/orders" element={<OrderHistoryPage />} />
                  <Route path="/admin" element={<AdminDashboard />} />
                  <Route path="*" element={<Home />} />
                </Routes>
              </main>

              <Footer />
              <ConsentBanner />
              <AuthModal
                isOpen={authModalOpen}
                onClose={() => setAuthModalOpen(false)}
              />
            </div>
          </CartWishlistProvider>
        </AuthProvider>
      </ToastProvider>
    </BrowserRouter>
  );
}

import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ToastProvider } from './context/ToastContext';
import { AuthProvider } from './context/AuthContext';
import { CartWishlistProvider } from './context/CartWishlistContext';

import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ConsentBanner from './components/ConsentBanner';
import AuthModal from './components/AuthModal';
import ErrorBoundary from './components/ErrorBoundary';

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
    <ErrorBoundary>
      <BrowserRouter>
        <ToastProvider>
          <AuthProvider>
            <CartWishlistProvider>
              <div className="min-h-screen flex flex-col bg-slate-50/50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 antialiased">
                <Navbar onOpenAuthModal={() => setAuthModalOpen(true)} />
                
                <main className="flex-1">
                  <Routes>
                    <Route path="/" element={<ErrorBoundary><Home /></ErrorBoundary>} />
                    <Route path="/products" element={<ErrorBoundary><ProductsPage /></ErrorBoundary>} />
                    <Route path="/products/:id" element={<ErrorBoundary><ProductDetailPage /></ErrorBoundary>} />
                    <Route path="/cart" element={<ErrorBoundary><CartPage /></ErrorBoundary>} />
                    <Route path="/wishlist" element={<ErrorBoundary><WishlistPage /></ErrorBoundary>} />
                    <Route path="/orders" element={<ErrorBoundary><OrderHistoryPage /></ErrorBoundary>} />
                    <Route path="/admin" element={<ErrorBoundary><AdminDashboard /></ErrorBoundary>} />
                    <Route path="*" element={<ErrorBoundary><Home /></ErrorBoundary>} />
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
    </ErrorBoundary>
  );
}

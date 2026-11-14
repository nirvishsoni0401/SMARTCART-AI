import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import { useAuth } from './AuthContext';
import { useToast } from './ToastContext';

export const CartWishlistContext = createContext(null);

export function CartWishlistProvider({ children }) {
  const [cart, setCart] = useState({ items: [], total_items: 0, subtotal: 0.0 });
  const [wishlist, setWishlist] = useState({ items: [], total_items: 0 });
  const [loading, setLoading] = useState(false);
  const { user } = useAuth();
  const { addToast } = useToast();

  const getSessionId = () => localStorage.getItem('smartcart_session_id') || 'guest_sess';

  const refreshCart = useCallback(async () => {
    try {
      const res = await api.getCart(getSessionId());
      setCart(res.data);
    } catch (e) {
      console.error('Failed to load cart:', e);
    }
  }, []);

  const refreshWishlist = useCallback(async () => {
    try {
      const res = await api.getWishlist(getSessionId());
      setWishlist(res.data);
    } catch (e) {
      console.error('Failed to load wishlist:', e);
    }
  }, []);

  useEffect(() => {
    refreshCart();
    refreshWishlist();
  }, [user, refreshCart, refreshWishlist]);

  const addToCart = async (productId, quantity = 1, showToast = true) => {
    try {
      const res = await api.addToCart(productId, quantity, getSessionId());
      setCart(res.data);
      if (showToast) addToast('Added to cart!', 'success');
      return true;
    } catch (e) {
      addToast('Failed to add to cart.', 'error');
      return false;
    }
  };

  const updateCartQty = async (itemId, quantity) => {
    try {
      const res = await api.updateCartItem(itemId, quantity, getSessionId());
      setCart(res.data);
    } catch (e) {
      addToast('Failed to update quantity.', 'error');
    }
  };

  const removeFromCart = async (itemId) => {
    try {
      const res = await api.removeCartItem(itemId, getSessionId());
      setCart(res.data);
      addToast('Removed from cart.', 'info');
    } catch (e) {
      addToast('Failed to remove item.', 'error');
    }
  };

  const clearCart = async () => {
    try {
      const res = await api.clearCart(getSessionId());
      setCart(res.data);
    } catch (e) {
      console.error('Failed to clear cart:', e);
    }
  };

  const addToWishlist = async (productId) => {
    try {
      const res = await api.addToWishlist(productId, getSessionId());
      setWishlist(res.data);
      addToast('Saved to wishlist!', 'success');
    } catch (e) {
      addToast('Failed to save to wishlist.', 'error');
    }
  };

  const removeFromWishlist = async (productId) => {
    try {
      const res = await api.removeFromWishlist(productId, getSessionId());
      setWishlist(res.data);
      addToast('Removed from wishlist.', 'info');
    } catch (e) {
      addToast('Failed to remove from wishlist.', 'error');
    }
  };

  const inWishlist = (productId) => {
    return wishlist.items.some((item) => item.product_id === productId);
  };

  return (
    <CartWishlistContext.Provider
      value={{
        cart,
        wishlist,
        addToCart,
        updateCartQty,
        removeFromCart,
        clearCart,
        addToWishlist,
        removeFromWishlist,
        inWishlist,
        refreshCart,
        refreshWishlist,
      }}
    >
      {children}
    </CartWishlistContext.Provider>
  );
}

export function useCartWishlist() {
  const context = useContext(CartWishlistContext);
  if (!context) {
    throw new Error('useCartWishlist must be used within a CartWishlistProvider');
  }
  return context;
}


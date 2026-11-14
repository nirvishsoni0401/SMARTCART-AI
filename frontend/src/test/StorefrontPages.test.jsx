import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import ProductCard from '../components/ProductCard';
import ErrorBoundary from '../components/ErrorBoundary';
import CartPage from '../pages/CartPage';
import WishlistPage from '../pages/WishlistPage';
import OrderHistoryPage from '../pages/OrderHistoryPage';
import { CartWishlistProvider, CartWishlistContext } from '../context/CartWishlistContext';
import { AuthProvider, AuthContext } from '../context/AuthContext';
import { ToastProvider } from '../context/ToastContext';
import { api } from '../api/client';

const mockProduct = {
  id: 101,
  name: 'Running Shoes Turbo',
  category: 'Sports',
  brand: 'Puma',
  price: 2499,
  mrp: 3999,
  discount_pct: 37.5,
  rating: 4.6,
  rating_count: 85,
  stock: 12,
  image_url: 'https://cdn.dummyjson.com/products/images/sports/1.png',
};

describe('ProductCard Recommendation Explanation Chip', () => {
  it('renders explanation chip when explanation prop is supplied', () => {
    render(
      <MemoryRouter>
        <ToastProvider>
          <AuthProvider>
            <CartWishlistProvider>
              <ProductCard
                product={mockProduct}
                explanation="Because you viewed Sports shoes"
                score={0.92}
              />
            </CartWishlistProvider>
          </AuthProvider>
        </ToastProvider>
      </MemoryRouter>
    );

    expect(screen.getByText('Because you viewed Sports shoes')).toBeInTheDocument();
    expect(screen.getByText('Running Shoes Turbo')).toBeInTheDocument();
    expect(screen.getByText('Puma')).toBeInTheDocument();
  });
});

describe('CartPage Rendering with Images', () => {
  it('renders cart items, product images, and order totals', () => {
    const mockCartValue = {
      cart: {
        items: [
          {
            id: 'item-1',
            quantity: 2,
            product: mockProduct,
          },
        ],
        total_items: 2,
        subtotal: 4998,
      },
      updateCartQty: vi.fn(),
      removeFromCart: vi.fn(),
      clearCart: vi.fn(),
      wishlist: { items: [], total_items: 0 },
      addToWishlist: vi.fn(),
      removeFromWishlist: vi.fn(),
      inWishlist: vi.fn().mockReturnValue(false),
      addToCart: vi.fn(),
    };

    const mockAuthValue = {
      user: { id: 1, name: 'Rahul Sharma', email: 'rahul@example.com' },
      loading: false,
      consentTracking: true,
      isAdmin: false,
    };

    render(
      <MemoryRouter>
        <ToastProvider>
          <AuthContext.Provider value={mockAuthValue}>
            <CartWishlistContext.Provider value={mockCartValue}>
              <CartPage />
            </CartWishlistContext.Provider>
          </AuthContext.Provider>
        </ToastProvider>
      </MemoryRouter>
    );

    expect(screen.getByText(/Shopping Cart \(2 items\)/i)).toBeInTheDocument();
    expect(screen.getByText('Running Shoes Turbo')).toBeInTheDocument();
    const imgs = screen.getAllByRole('img');
    expect(imgs.length).toBeGreaterThan(0);
    expect(screen.getByText(/Grand Total/i)).toBeInTheDocument();
  });
});

describe('WishlistPage Rendering with Images', () => {
  it('renders wishlist items and images', () => {
    const mockWishlistValue = {
      cart: { items: [], total_items: 0, subtotal: 0 },
      wishlist: {
        items: [
          {
            id: 'wish-1',
            product: mockProduct,
          },
        ],
        total_items: 1,
      },
      updateCartQty: vi.fn(),
      removeFromCart: vi.fn(),
      clearCart: vi.fn(),
      addToWishlist: vi.fn(),
      removeFromWishlist: vi.fn(),
      inWishlist: vi.fn().mockReturnValue(true),
      addToCart: vi.fn(),
    };

    render(
      <MemoryRouter>
        <ToastProvider>
          <AuthProvider>
            <CartWishlistContext.Provider value={mockWishlistValue}>
              <WishlistPage />
            </CartWishlistContext.Provider>
          </AuthProvider>
        </ToastProvider>
      </MemoryRouter>
    );

    expect(screen.getByText(/Your Wishlist \(1 items\)/i)).toBeInTheDocument();
    expect(screen.getByText('Running Shoes Turbo')).toBeInTheDocument();
    expect(screen.getByRole('img')).toBeInTheDocument();
  });
});

describe('OrderHistoryPage Rendering with Images', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders order history with product images', async () => {
    const mockUser = { id: 1, name: 'Rahul Sharma', email: 'rahul@example.com' };
    api.getOrders.mockResolvedValueOnce({
      data: [
        {
          id: 501,
          created_at: '2026-11-10T12:00:00Z',
          status: 'delivered',
          total_amount: 2499,
          items: [
            {
              id: 'ord-item-1',
              product_id: 101,
              quantity: 1,
              price: 2499,
              is_returned: false,
              product: mockProduct,
            },
          ],
        },
      ],
    });

    render(
      <MemoryRouter>
        <ToastProvider>
          <AuthContext.Provider value={{ user: mockUser, loading: false }}>
            <OrderHistoryPage />
          </AuthContext.Provider>
        </ToastProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText('Running Shoes Turbo')).toBeInTheDocument();
    });

    expect(screen.getByText(/Order ID:/i)).toBeInTheDocument();
    expect(screen.getByRole('img')).toBeInTheDocument();
  });
});

describe('ErrorBoundary Resilience', () => {
  it('catches render errors and displays friendly fallback with reload button and back to shop link', () => {
    const ThrowingComponent = () => {
      throw new Error('Test crash in child component');
    };

    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText(/We encountered an unexpected error/i)).toBeInTheDocument();
    expect(screen.getByText('Test crash in child component')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Reload/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to shop/i })).toBeInTheDocument();

    spy.mockRestore();
  });
});

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import ProductDetailPage from '../pages/ProductDetailPage';
import { ToastProvider } from '../context/ToastContext';
import { CartWishlistProvider } from '../context/CartWishlistContext';
import { AuthProvider } from '../context/AuthContext';
import { api } from '../api/client';

const mockProduct = {
  id: 758,
  name: 'Clean Architecture in Python',
  category: 'Books',
  brand: 'O\'Reilly',
  price: 899,
  mrp: 1299,
  discount_pct: 30.79,
  rating: 4.8,
  rating_count: 142,
  stock: 25,
  description: 'A comprehensive guide to software architecture principles using Python.',
  image_url: '/static/covers/book_758_test.svg',
  attributes: {
    gallery: [
      'https://cdn.dummyjson.com/products/images/books/1.png',
      'https://cdn.dummyjson.com/products/images/books/2.png'
    ],
    shippingInformation: 'Ships in 1-2 days',
    warrantyInformation: 'Publisher guarantee',
    returnPolicy: '7-Day Return',
  },
  reviews: [
    {
      id: 1,
      reviewer_name: 'Amit Patel',
      rating: 5,
      comment: 'Superb read, transformed how I structure modules.',
      date: '2026-10-15',
    }
  ]
};

function renderWithProviders(initialRoute = '/products/758') {
  return render(
    <MemoryRouter initialEntries={[initialRoute]}>
      <ToastProvider>
        <AuthProvider>
          <CartWishlistProvider>
            <Routes>
              <Route path="/products/:id" element={<ProductDetailPage />} />
              <Route path="/products" element={<div>Catalog Page</div>} />
            </Routes>
          </CartWishlistProvider>
        </AuthProvider>
      </ToastProvider>
    </MemoryRouter>
  );
}

describe('ProductDetailPage Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading skeleton then transitions to loaded product without hook order crash', async () => {
    api.getProduct.mockResolvedValueOnce({ data: mockProduct });
    api.getSimilarProducts.mockResolvedValueOnce({ data: [] });

    renderWithProviders('/products/758');

    // Initially in loading state (11 hooks mounted)
    expect(screen.getByTestId('product-loading')).toBeInTheDocument();

    // Resolves and transitions to loaded view (where previously hook #12 was conditionally called)
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Clean Architecture in Python', level: 1 })).toBeInTheDocument();
    });

    expect(screen.getByText(/₹899/)).toBeInTheDocument();
    expect(screen.getByText(/O'Reilly/i)).toBeInTheDocument();
  });

  it('renders product not found gracefully when product is null or inactive', async () => {
    api.getProduct.mockRejectedValueOnce(new Error('Product not found'));
    api.getSimilarProducts.mockResolvedValueOnce({ data: [] });

    renderWithProviders('/products/9999');

    await waitFor(() => {
      expect(screen.getByTestId('product-not-found')).toBeInTheDocument();
    });
    expect(screen.getByText(/Product not found or inactive/i)).toBeInTheDocument();
    expect(screen.getByText(/Back to Catalog/i)).toBeInTheDocument();
  });

  it('renders gallery thumbnails and cycles through them using keyboard arrows', async () => {
    api.getProduct.mockResolvedValueOnce({ data: mockProduct });
    api.getSimilarProducts.mockResolvedValueOnce({ data: [] });

    renderWithProviders('/products/758');

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Clean Architecture in Python', level: 1 })).toBeInTheDocument();
    });

    // Verify thumbnail tablist exists
    const tablist = screen.getByRole('tablist', { name: /Product image thumbnails/i });
    expect(tablist).toBeInTheDocument();

    const tabs = screen.getAllByRole('tab');
    expect(tabs.length).toBeGreaterThan(1);

    // Initial thumbnail selected
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true');

    // Arrow navigation: press ArrowRight on the first thumbnail
    fireEvent.keyDown(tabs[0], { key: 'ArrowRight' });
    expect(tabs[1]).toHaveAttribute('aria-selected', 'true');

    // Arrow navigation: press ArrowLeft to go back
    fireEvent.keyDown(tabs[1], { key: 'ArrowLeft' });
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true');
  });

  it('opens zoom modal on main image click and closes on Escape key press', async () => {
    const user = userEvent.setup();
    api.getProduct.mockResolvedValueOnce({ data: mockProduct });
    api.getSimilarProducts.mockResolvedValueOnce({ data: [] });

    renderWithProviders('/products/758');

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Clean Architecture in Python', level: 1 })).toBeInTheDocument();
    });

    // Modal is initially not present
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();

    // Click the zoom trigger button
    const zoomTrigger = screen.getByLabelText(/Click to enlarge product photo/i);
    await user.click(zoomTrigger);

    // Lightbox modal should now be visible
    expect(screen.getByRole('dialog', { name: /Enlarged product image preview/i })).toBeInTheDocument();

    // Press Escape to dismiss modal
    fireEvent.keyDown(window, { key: 'Escape' });

    // Modal should close
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });
});

import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Polyfill window.matchMedia if not implemented in jsdom
if (typeof window !== 'undefined' && !window.matchMedia) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

// Polyfill window.scrollTo
if (typeof window !== 'undefined' && !window.scrollTo) {
  window.scrollTo = () => {};
}

// Mock api client with complete method stubs for all tests
vi.mock('../api/client', () => ({
  api: {
    getProduct: vi.fn().mockResolvedValue({ data: null }),
    getSimilarProducts: vi.fn().mockResolvedValue({ data: [] }),
    getProducts: vi.fn().mockResolvedValue({ data: { products: [], total: 0 } }),
    getProductCount: vi.fn().mockResolvedValue({ data: { total: 262 } }),
    getCart: vi.fn().mockResolvedValue({ data: { items: [], total_items: 0, subtotal: 0 } }),
    getWishlist: vi.fn().mockResolvedValue({ data: { items: [], total_items: 0 } }),
    getOrders: vi.fn().mockResolvedValue({ data: [] }),
    returnItem: vi.fn().mockResolvedValue({}),
    recordEvent: vi.fn().mockResolvedValue({}),
    recordImpressions: vi.fn().mockResolvedValue({}),
    getRecentlyViewed: vi.fn().mockResolvedValue({ data: [] }),
    getMe: vi.fn().mockResolvedValue({ data: { id: 1, name: 'Test User', email: 'test@smartcart.ai' } }),
    login: vi.fn().mockResolvedValue(true),
    register: vi.fn().mockResolvedValue(true),
  },
}));

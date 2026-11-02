import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Attach Authorization token and session ID
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('smartcart_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  let sessionId = localStorage.getItem('smartcart_session_id');
  if (!sessionId) {
    sessionId = 'sess_' + Math.random().toString(36).substring(2, 12);
    localStorage.setItem('smartcart_session_id', sessionId);
  }
  config.headers['x-session-id'] = sessionId;
  return config;
});

export const api = {
  // Auth
  register: (data) => apiClient.post('/auth/register', data),
  login: (data) => apiClient.post('/auth/login', data),
  getMe: () => apiClient.get('/auth/me'),
  updateConsent: (consent) => apiClient.put('/auth/consent', { consent_tracking: consent }),

  // Products
  getProducts: (params) => apiClient.get('/products', { params }),
  getCategories: () => apiClient.get('/products/categories'),
  getBrands: (category) => apiClient.get('/products/brands', { params: { category } }),
  getProduct: (id) => apiClient.get(`/products/${id}`),
  getSimilarProducts: (id) => apiClient.get(`/products/${id}/similar`),

  // Cart
  getCart: (sessionId) => apiClient.get('/cart', { params: { session_id: sessionId } }),
  addToCart: (productId, quantity = 1, sessionId) =>
    apiClient.post('/cart', { product_id: productId, quantity, session_id: sessionId }),
  updateCartItem: (itemId, quantity, sessionId) =>
    apiClient.put(`/cart/${itemId}`, { quantity }, { params: { session_id: sessionId } }),
  removeCartItem: (itemId, sessionId) =>
    apiClient.delete(`/cart/${itemId}`, { params: { session_id: sessionId } }),
  clearCart: (sessionId) => apiClient.delete('/cart', { params: { session_id: sessionId } }),

  // Wishlist
  getWishlist: (sessionId) => apiClient.get('/wishlist', { params: { session_id: sessionId } }),
  addToWishlist: (productId, sessionId) =>
    apiClient.post('/wishlist', { product_id: productId, session_id: sessionId }),
  removeFromWishlist: (productId, sessionId) =>
    apiClient.delete(`/wishlist/${productId}`, { params: { session_id: sessionId } }),

  // Orders
  checkout: (data) => apiClient.post('/orders', data),
  getOrders: () => apiClient.get('/orders'),
  getOrder: (id) => apiClient.get(`/orders/${id}`),
  returnItem: (orderId, itemId, reason = 'Not satisfied') =>
    apiClient.post(`/orders/${orderId}/items/${itemId}/return`, { reason }),

  // Recommendations
  getRecommendations: (params) => apiClient.get('/recommendations', { params }),

  // Events
  recordEvent: (productId, eventType, logId, sessionId) =>
    apiClient.post('/events', {
      product_id: productId,
      event_type: eventType,
      recommendation_log_id: logId,
      session_id: sessionId,
    }),
  recordImpressions: (productIds, logIds, sessionId) =>
    apiClient.post('/events/impressions', {
      product_ids: productIds,
      recommendation_log_ids: logIds,
      session_id: sessionId,
    }),

  // Admin
  getMetrics: (window = 'all') => apiClient.get('/admin/metrics', { params: { window } }),
  getAlgorithms: () => apiClient.get('/admin/algorithms'),
  setAlgorithm: (algorithm) => apiClient.put('/admin/algorithms', { algorithm }),
  getRewardConfig: () => apiClient.get('/admin/reward-config'),
  updateRewardConfig: (data) => apiClient.put('/admin/reward-config', data),
  resetModels: () => apiClient.post('/admin/reset-models'),
  runSimulate: (numRounds = 50) => apiClient.post('/admin/simulate', { num_rounds: numRounds }),
  getLogs: (limit = 25, algorithm) =>
    apiClient.get('/admin/logs', { params: { limit, algorithm } }),
};


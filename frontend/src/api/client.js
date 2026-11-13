import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Attach Authorization token automatically if present
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('smartcart_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
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
  getProduct: (id) => apiClient.get(`/products/${id}`),
  getSimilarProducts: (id) => apiClient.get(`/products/${id}/similar`),
  getCategories: () => apiClient.get('/products/categories'),
  getBrands: (category) => apiClient.get('/products/brands', { params: { category } }),

  // Cart
  getCart: (sessionId) => apiClient.get('/cart', { params: { session_id: sessionId } }),
  addToCart: (productId, quantity = 1, sessionId) =>
    apiClient.post('/cart', { product_id: productId, quantity, session_id: sessionId }),
  updateCartItem: (itemId, quantity, sessionId) =>
    apiClient.put(`/cart/items/${itemId}`, { quantity }, { params: { session_id: sessionId } }),
  removeCartItem: (itemId, sessionId) =>
    apiClient.delete(`/cart/items/${itemId}`, { params: { session_id: sessionId } }),
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
  returnOrderItem: (orderId, itemId, reason) =>
    apiClient.post(`/orders/${orderId}/items/${itemId}/return`, { reason }),

  // Recommendations
  getRecommendations: (params) => apiClient.get('/recommendations', { params }),
  getRecentlyViewed: (limit = 10, sessionId) =>
    apiClient.get('/events/recently-viewed', { params: { limit, session_id: sessionId } }),

  // Feedback Events
  recordEvent: (productId, eventType, logId, sessionId, origin = 'live') =>
    apiClient.post('/events', {
      product_id: productId,
      event_type: eventType,
      recommendation_log_id: logId,
      session_id: sessionId,
      origin: origin,
    }),
  recordImpressions: (productIds, logIds, sessionId, origin = 'live') =>
    apiClient.post('/events/impressions', {
      product_ids: productIds,
      recommendation_log_ids: logIds,
      session_id: sessionId,
      origin: origin,
    }),

  // Admin
  getMetrics: (window = 'all', origin = 'live') =>
    apiClient.get('/admin/metrics', { params: { window, origin } }),
  getAlgorithms: () => apiClient.get('/admin/algorithms'),
  setAlgorithm: (algorithm) => apiClient.put('/admin/algorithms', { algorithm }),
  getRewardConfig: () => apiClient.get('/admin/reward-config'),
  updateRewardConfig: (data) => apiClient.put('/admin/reward-config', data),
  resetModels: (sandboxOnly = false) =>
    apiClient.post('/admin/reset-models', null, { params: { sandbox_only: sandboxOnly } }),
  runSimulate: (numRounds = 50, trainLive = false) =>
    apiClient.post('/admin/simulate', { num_rounds: numRounds, train_live: trainLive }),
  getLogs: (limit = 25, algorithm, origin = 'live') =>
    apiClient.get('/admin/logs', { params: { limit, algorithm, origin } }),
  createModelSnapshot: (name = 'pre_phase8') =>
    apiClient.post('/admin/models/snapshot', null, { params: { name } }),
  rollbackModelSnapshot: (name = 'pre_phase8') =>
    apiClient.post('/admin/models/rollback', null, { params: { name } }),
};

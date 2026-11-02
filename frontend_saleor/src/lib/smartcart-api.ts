/**
 * SmartCart AI Centralized API Client
 * Connects Saleor Storefront components to FastAPI + MongoDB Atlas backend
 * Supports 10,000+ catalog, Indian Rupee (₹) pricing, and LinUCB bandits.
 */

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export interface SmartCartProduct {
  product_id: number;
  external_id: number;
  title: string;
  name: string;
  description: string;
  category: string;
  category_slug?: string;
  subcategory?: string;
  subcategory_slug?: string;
  brand: string;
  price: number;
  discounted_price: number;
  discount_percentage?: number;
  currency: string;
  rating: number;
  review_count?: number;
  stock_quantity: number;
  stock: number;
  availability: string;
  image_url: string;
  thumbnail: string;
  images: string[];
  seller?: string;
  specifications?: Record<string, string>;
  tags?: string[];
  popularity_score?: number;
  is_active?: boolean;
}

export interface ProductsResponse {
  total: number;
  limit: number;
  skip: number;
  page?: number;
  total_pages?: number;
  products: SmartCartProduct[];
}

export interface CategoryDetail {
  name: string;
  slug: string;
  count: number;
  subcategories?: string[];
}

export interface CategoriesResponse {
  categories: string[];
  details: CategoryDetail[];
}

export interface SearchSuggestionsResponse {
  query: string;
  brands: string[];
  subcategories: string[];
  products: {
    product_id: number;
    title: string;
    category: string;
    thumbnail: string;
    price: number;
  }[];
}

export interface FilterFacetsResponse {
  category?: string;
  brands: string[];
  subcategories: string[];
  price_range: {
    min: number;
    max: number;
  };
  rating_options: number[];
  discount_options: number[];
}

export interface RecommendationItem {
  product_id: number;
  external_id: number;
  title: string;
  name: string;
  category: string;
  subcategory?: string;
  brand: string;
  price: number;
  discounted_price: number;
  discount_percentage: number;
  currency: string;
  rating: number;
  review_count?: number;
  image_url: string;
  thumbnail: string;
  images?: string[];
  stock: number;
  stock_quantity: number;
  availability?: string;
  explanation: string;
  ucb_score: number;
  expected_reward: number;
  exploration_bonus: number;
  arm_id: string;
}

export interface RecommendationsResponse {
  recommendation_id: string;
  algorithm: string;
  context_summary: {
    dominant_category?: string;
    budget_level?: string;
    interactions?: number;
  };
  recommendations: RecommendationItem[];
}

export interface CartItem {
  product_id: number;
  quantity: number;
  name: string;
  price: number;
  currency: string;
  image_url: string;
  category: string;
  brand: string;
  stock: number;
  in_stock: boolean;
  total_price: number;
}

export interface CartResponse {
  items: CartItem[];
  total_items: number;
  subtotal: number;
  currency: string;
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: string;
  preferred_categories: string[];
  created_at: string;
}

/** Formats a numeric price into standard Indian Rupee notation (e.g. ₹1,49,999) */
export function formatINR(amount: number): string {
  if (typeof amount !== "number" || isNaN(amount)) return "₹0";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(amount);
}

class SmartCartClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private getAuthHeader(): Record<string, string> {
    if (typeof window !== "undefined") {
      const token = localStorage.getItem("smartcart_token");
      if (token) {
        return { Authorization: `Bearer ${token}` };
      }
    }
    return {};
  }

  private getSessionHeader(): Record<string, string> {
    if (typeof window !== "undefined") {
      let sessionId = localStorage.getItem("smartcart_session_id");
      if (!sessionId) {
        sessionId = "sess_" + Math.random().toString(36).substring(2, 12);
        localStorage.setItem("smartcart_session_id", sessionId);
      }
      return { "x-session-id": sessionId };
    }
    return {};
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const headers = {
      "Content-Type": "application/json",
      ...this.getAuthHeader(),
      ...this.getSessionHeader(),
      ...(options.headers || {}),
    };

    try {
      const res = await fetch(url, {
        ...options,
        headers,
        credentials: "include",
      });

      if (!res.ok) {
        let errDetail = `HTTP ${res.status}: ${res.statusText}`;
        try {
          const errData = await res.json();
          errDetail = errData.detail || errDetail;
        } catch {}
        throw new Error(errDetail);
      }

      return (await res.json()) as T;
    } catch (err: any) {
      console.error(`[SmartCart API] Request to ${url} failed:`, err.message);
      throw err;
    }
  }

  // --- Products Catalog ---

  async getProducts(params?: {
    category?: string;
    subcategory?: string;
    brand?: string;
    min_price?: number;
    max_price?: number;
    min_rating?: number;
    min_discount?: number;
    in_stock_only?: boolean;
    sort_by?: string;
    limit?: number;
    skip?: number;
    page?: number;
  }): Promise<ProductsResponse> {
    const query = new URLSearchParams();
    if (params?.category) query.set("category", params.category);
    if (params?.subcategory) query.set("subcategory", params.subcategory);
    if (params?.brand) query.set("brand", params.brand);
    if (params?.min_price != null) query.set("min_price", String(params.min_price));
    if (params?.max_price != null) query.set("max_price", String(params.max_price));
    if (params?.min_rating != null) query.set("min_rating", String(params.min_rating));
    if (params?.min_discount != null) query.set("min_discount", String(params.min_discount));
    if (params?.in_stock_only) query.set("in_stock_only", "true");
    if (params?.sort_by) query.set("sort_by", params.sort_by);
    if (params?.limit != null) query.set("limit", String(params.limit));
    if (params?.skip != null) query.set("skip", String(params.skip));
    if (params?.page != null) query.set("page", String(params.page));

    const qs = query.toString();
    return this.request<ProductsResponse>(`/products/${qs ? `?${qs}` : ""}`, {
      next: { revalidate: 30 },
    });
  }

  async getProductById(id: string | number): Promise<SmartCartProduct> {
    return this.request<SmartCartProduct>(`/products/${id}`, {
      next: { revalidate: 60 },
    });
  }

  async getCategories(): Promise<CategoriesResponse> {
    return this.request<CategoriesResponse>("/products/categories", {
      next: { revalidate: 300 },
    });
  }

  async getSubcategories(category?: string): Promise<{ subcategories: string[] }> {
    const qs = category ? `?category=${encodeURIComponent(category)}` : "";
    return this.request<{ subcategories: string[] }>(`/products/subcategories${qs}`);
  }

  async getBrands(category?: string, subcategory?: string): Promise<{ brands: string[] }> {
    const query = new URLSearchParams();
    if (category) query.set("category", category);
    if (subcategory) query.set("subcategory", subcategory);
    const qs = query.toString();
    return this.request<{ brands: string[] }>(`/products/brands${qs ? `?${qs}` : ""}`);
  }

  async getFilterFacets(category?: string): Promise<FilterFacetsResponse> {
    const qs = category ? `?category=${encodeURIComponent(category)}` : "";
    return this.request<FilterFacetsResponse>(`/products/filters${qs}`);
  }

  // --- Curated Homepage Sections ---

  async getTrendingProducts(limit: number = 12): Promise<{ products: SmartCartProduct[] }> {
    return this.request<{ products: SmartCartProduct[] }>(`/products/trending?limit=${limit}`, {
      next: { revalidate: 60 },
    });
  }

  async getBestsellers(limit: number = 12): Promise<{ products: SmartCartProduct[] }> {
    return this.request<{ products: SmartCartProduct[] }>(`/products/bestsellers?limit=${limit}`, {
      next: { revalidate: 60 },
    });
  }

  async getTodaysDeals(limit: number = 12): Promise<{ products: SmartCartProduct[] }> {
    return this.request<{ products: SmartCartProduct[] }>(`/products/deals?limit=${limit}`, {
      next: { revalidate: 60 },
    });
  }

  async getTopRated(limit: number = 12): Promise<{ products: SmartCartProduct[] }> {
    return this.request<{ products: SmartCartProduct[] }>(`/products/top-rated?limit=${limit}`, {
      next: { revalidate: 60 },
    });
  }

  async getRecentlyAdded(limit: number = 12): Promise<{ products: SmartCartProduct[] }> {
    return this.request<{ products: SmartCartProduct[] }>(`/products/recently-added?limit=${limit}`, {
      next: { revalidate: 60 },
    });
  }

  // --- Search & Autocomplete ---

  async getSearchSuggestions(query: string): Promise<SearchSuggestionsResponse> {
    return this.request<SearchSuggestionsResponse>(`/products/search/suggestions?q=${encodeURIComponent(query)}`);
  }

  async searchProducts(
    query: string,
    limit: number = 24,
    skip: number = 0,
    page?: number,
    sort_by?: string
  ): Promise<ProductsResponse & { query: string }> {
    const params = new URLSearchParams({ q: query, limit: String(limit), skip: String(skip) });
    if (page) params.set("page", String(page));
    if (sort_by) params.set("sort_by", sort_by);
    return this.request<ProductsResponse & { query: string }>(`/products/search?${params.toString()}`);
  }

  // --- PDP Cross-Sell ---

  async getSimilarProducts(id: string | number, limit: number = 6): Promise<{ products: SmartCartProduct[] }> {
    return this.request<{ products: SmartCartProduct[] }>(`/products/${id}/similar?limit=${limit}`);
  }

  async getFrequentlyBoughtTogether(id: string | number, limit: number = 4): Promise<{ products: SmartCartProduct[] }> {
    return this.request<{ products: SmartCartProduct[] }>(`/products/${id}/frequently-bought-together?limit=${limit}`);
  }

  // --- LinUCB Recommendations ---

  async getRecommendations(limit: number = 8): Promise<RecommendationsResponse> {
    return this.request<RecommendationsResponse>(`/recommendations/?limit=${limit}`, {
      cache: "no-store",
    });
  }

  async sendRecommendationFeedback(payload: {
    recommendation_id?: string;
    product_id: number;
    action: "like" | "dislike" | "click" | "cart" | "purchase" | "not_interested";
  }): Promise<any> {
    return this.request("/recommendations/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  // --- Behaviour Tracking ---

  async trackEvent(payload: {
    event_type: string;
    product_id?: number;
    recommendation_id?: string;
    metadata?: Record<string, any>;
  }): Promise<any> {
    return this.request("/events/", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  // --- Cart ---

  async getCart(): Promise<CartResponse> {
    return this.request<CartResponse>("/cart/", { cache: "no-store" });
  }

  async addToCart(productId: number, quantity: number = 1): Promise<any> {
    return this.request("/cart/add", {
      method: "POST",
      body: JSON.stringify({ product_id: productId, quantity }),
    });
  }

  async updateCart(productId: number, quantity: number): Promise<any> {
    return this.request("/cart/update", {
      method: "PUT",
      body: JSON.stringify({ product_id: productId, quantity }),
    });
  }

  async removeFromCart(productId: number): Promise<any> {
    return this.request(`/cart/${productId}`, {
      method: "DELETE",
    });
  }

  // --- Wishlist ---

  async getWishlist(): Promise<{ items: any[]; total_items: number }> {
    return this.request("/wishlist/", { cache: "no-store" });
  }

  async addToWishlist(productId: number): Promise<any> {
    return this.request("/wishlist/add", {
      method: "POST",
      body: JSON.stringify({ product_id: productId }),
    });
  }

  async removeFromWishlist(productId: number): Promise<any> {
    return this.request(`/wishlist/${productId}`, {
      method: "DELETE",
    });
  }

  // --- Orders & Checkout ---

  async createOrder(orderData: {
    shipping_address: {
      full_name: string;
      street_address: string;
      city: string;
      state: string;
      postal_code: string;
      country?: string;
    };
    customer_email?: string;
    customer_name?: string;
    payment_method?: string;
  }): Promise<any> {
    return this.request("/orders/", {
      method: "POST",
      body: JSON.stringify(orderData),
    });
  }

  async getOrders(): Promise<{ orders: any[]; total: number }> {
    return this.request("/orders/", { cache: "no-store" });
  }

  async getOrderById(orderId: string): Promise<{ order: any }> {
    return this.request(`/orders/${orderId}`);
  }

  // --- Authentication ---

  async register(data: { name: string; email: string; password: string; preferred_categories?: string[] }): Promise<any> {
    const res = await this.request<any>("/auth/register", {
      method: "POST",
      body: JSON.stringify(data),
    });
    if (res.access_token && typeof window !== "undefined") {
      localStorage.setItem("smartcart_token", res.access_token);
      localStorage.setItem("smartcart_user", JSON.stringify(res.user));
    }
    return res;
  }

  async login(email: string, password: string): Promise<any> {
    const res = await this.request<any>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    if (res.access_token && typeof window !== "undefined") {
      localStorage.setItem("smartcart_token", res.access_token);
      localStorage.setItem("smartcart_user", JSON.stringify(res.user));
    }
    return res;
  }

  async logout(): Promise<any> {
    if (typeof window !== "undefined") {
      localStorage.removeItem("smartcart_token");
      localStorage.removeItem("smartcart_user");
    }
    return this.request("/auth/logout", { method: "POST" });
  }

  async getMe(): Promise<{ user: UserProfile }> {
    return this.request<{ user: UserProfile }>("/auth/me");
  }

  // --- Admin ---

  async getAdminStats(): Promise<any> {
    return this.request("/admin/stats", { cache: "no-store" });
  }

  // --- Analytics & Research ---

  async getAnalyticsOverview(): Promise<any> {
    return this.request("/analytics/overview", { cache: "no-store" });
  }

  async getAlgorithmComparison(): Promise<any> {
    return this.request("/analytics/comparison", { cache: "no-store" });
  }

  async runSimulation(steps: number = 100, seed: number = 42): Promise<any> {
    return this.request("/analytics/simulate", {
      method: "POST",
      body: JSON.stringify({ steps, seed }),
    });
  }
}

export const smartCartApi = new SmartCartClient();

import React from 'react';
import { Link } from 'react-router-dom';
import { Heart, ShoppingBag, Trash2, ArrowRight } from 'lucide-react';
import { useCartWishlist } from '../context/CartWishlistContext';

export default function WishlistPage() {
  const { wishlist, removeFromWishlist, addToCart } = useCartWishlist();

  if (wishlist.items.length === 0) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center space-y-4">
        <div className="p-4 bg-rose-50 dark:bg-rose-950/60 text-rose-500 w-fit mx-auto rounded-full">
          <Heart className="w-10 h-10" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Your Wishlist is Empty</h2>
        <p className="text-xs text-slate-500 max-w-sm mx-auto">
          Save your favorite items here to track discounts, stock availability, and receive personalized recommendations.
        </p>
        <Link
          to="/products"
          className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-indigo-600 text-white text-xs font-bold shadow-md hover:bg-indigo-500 transition"
        >
          <span>Explore Products</span>
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    );
  }

  const handleMoveToCart = async (productId) => {
    await addToCart(productId, 1);
    await removeFromWishlist(productId);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <h1 className="text-2xl font-black text-slate-900 dark:text-white mb-8">
        Your Wishlist ({wishlist.total_items} items)
      </h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {wishlist.items.map((item) => (
          <div
            key={item.id}
            className="flex flex-col justify-between p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm"
          >
            <div>
              <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-100 dark:bg-slate-800 mb-3">
                <img
                  src={item.product.image_url}
                  alt={item.product.name}
                  className="w-full h-full object-cover"
                />
                <button
                  onClick={() => removeFromWishlist(item.product.id)}
                  className="absolute top-2.5 right-2.5 p-1.5 rounded-full bg-white/80 dark:bg-slate-900/80 text-rose-500 hover:bg-white shadow-sm"
                  title="Remove from wishlist"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>

              <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600">
                {item.product.brand}
              </span>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-white line-clamp-2 mb-2">
                <Link to={`/products/${item.product.id}`} className="hover:underline">
                  {item.product.name}
                </Link>
              </h3>
              <div className="text-base font-extrabold text-slate-900 dark:text-white mb-4">
                ₹{item.product.price.toLocaleString('en-IN')}
              </div>
            </div>

            <button
              onClick={() => handleMoveToCart(item.product.id)}
              className="w-full py-2 px-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold flex items-center justify-center gap-1.5 shadow-sm transition"
            >
              <ShoppingBag className="w-3.5 h-3.5" />
              <span>Move to Cart</span>
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}


import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Trash2, ShoppingBag, ArrowRight, ShieldCheck, Check } from 'lucide-react';
import { useCartWishlist } from '../context/CartWishlistContext';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../context/ToastContext';
import { api } from '../api/client';

export default function CartPage() {
  const { cart, updateCartQty, removeFromCart, clearCart } = useCartWishlist();
  const { user } = useAuth();
  const { addToast } = useToast();
  const navigate = useNavigate();

  const [shippingAddress, setShippingAddress] = useState({
    name: user?.name || 'Rahul Sharma',
    street: '123 MG Road, Koramangala',
    city: 'Bengaluru',
    state: 'Karnataka',
    pin: '560034',
  });
  const [checkingOut, setCheckingOut] = useState(false);
  const [showAddressModal, setShowAddressModal] = useState(false);

  const deliveryFee = cart.subtotal > 999 || cart.total_items === 0 ? 0 : 99;
  const grandTotal = cart.subtotal + deliveryFee;

  const handleCheckout = async (e) => {
    e.preventDefault();
    if (!user) {
      addToast('Please login to place an order.', 'info');
      return;
    }
    if (cart.items.length === 0) {
      addToast('Your cart is empty.', 'error');
      return;
    }

    setCheckingOut(true);
    const sessionId = localStorage.getItem('smartcart_session_id');

    try {
      const res = await api.checkout({
        shipping_address: shippingAddress,
        payment_method: 'UPI / Mock Pay',
        session_id: sessionId,
      });

      await clearCart();
      addToast(`Order placed successfully! Order #${res.data.id}`, 'success');
      navigate('/orders');
    } catch (err) {
      const msg = err.response?.data?.detail || 'Checkout failed. Please try again.';
      addToast(msg, 'error');
    } finally {
      setCheckingOut(false);
      setShowAddressModal(false);
    }
  };

  if (cart.items.length === 0) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center space-y-4">
        <div className="p-4 bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 w-fit mx-auto rounded-full">
          <ShoppingBag className="w-10 h-10" />
        </div>
        <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Your Cart is Empty</h2>
        <p className="text-xs text-slate-500 max-w-sm mx-auto">
          Explore our collection of 600+ top-rated products across 6 departments with personalized AI recommendations.
        </p>
        <Link
          to="/products"
          className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-indigo-600 text-white text-xs font-bold shadow-md hover:bg-indigo-500 transition"
        >
          <span>Start Shopping</span>
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <h1 className="text-2xl font-black text-slate-900 dark:text-white mb-8">
        Shopping Cart ({cart.total_items} items)
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10 items-start">
        {/* Items List */}
        <div className="lg:col-span-2 space-y-4">
          {cart.items.map((item) => (
            <div
              key={item.id}
              className="flex items-center gap-4 p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm"
            >
              <img
                src={item.product.image_url}
                alt={item.product.name}
                className="w-20 h-20 rounded-xl object-cover bg-slate-100 shrink-0"
              />

              <div className="flex-1 min-w-0">
                <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600">
                  {item.product.brand}
                </span>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white truncate">
                  <Link to={`/products/${item.product.id}`} className="hover:underline">
                    {item.product.name}
                  </Link>
                </h3>
                <div className="text-xs font-bold text-slate-900 dark:text-white mt-1">
                  ₹{item.product.price.toLocaleString('en-IN')}
                </div>
              </div>

              {/* Quantity Stepper */}
              <div className="flex items-center border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden text-xs font-semibold">
                <button
                  onClick={() => updateCartQty(item.id, item.quantity - 1)}
                  className="px-2.5 py-1.5 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200"
                >
                  -
                </button>
                <span className="px-3 py-1.5 min-w-[2rem] text-center">{item.quantity}</span>
                <button
                  onClick={() => updateCartQty(item.id, item.quantity + 1)}
                  className="px-2.5 py-1.5 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200"
                >
                  +
                </button>
              </div>

              {/* Item Total */}
              <div className="text-sm font-extrabold text-slate-900 dark:text-white shrink-0 min-w-[5rem] text-right">
                ₹{(item.product.price * item.quantity).toLocaleString('en-IN')}
              </div>

              {/* Remove */}
              <button
                onClick={() => removeFromCart(item.id)}
                className="p-2 text-slate-400 hover:text-rose-500 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>

        {/* Order Summary Card */}
        <div className="p-6 rounded-3xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-4">
          <h2 className="text-base font-bold text-slate-900 dark:text-white pb-3 border-b border-slate-200 dark:border-slate-800">
            Order Summary
          </h2>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between text-slate-500">
              <span>Items Subtotal</span>
              <span className="font-semibold text-slate-900 dark:text-white">
                ₹{cart.subtotal.toLocaleString('en-IN')}
              </span>
            </div>
            <div className="flex justify-between text-slate-500">
              <span>Delivery Fee</span>
              <span className="font-semibold text-emerald-600">
                {deliveryFee === 0 ? 'FREE' : `₹${deliveryFee}`}
              </span>
            </div>
            <div className="border-t border-slate-200 dark:border-slate-800 pt-2 flex justify-between text-sm font-black text-slate-900 dark:text-white">
              <span>Grand Total</span>
              <span>₹{grandTotal.toLocaleString('en-IN')}</span>
            </div>
          </div>

          {/* Delivery Note */}
          <div className="p-3 bg-indigo-50/50 dark:bg-indigo-950/30 rounded-xl border border-indigo-100 dark:border-indigo-900/40 text-[11px] text-slate-600 dark:text-slate-400 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-indigo-500 shrink-0" />
            <span>Eligible for 7-day hassle-free item returns</span>
          </div>

          <button
            onClick={() => setShowAddressModal(true)}
            className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold shadow-md transition flex items-center justify-center gap-2"
          >
            <span>Proceed to Checkout</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Checkout / Shipping Address Modal */}
      {showAddressModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-md bg-white dark:bg-slate-900 rounded-3xl p-6 border border-slate-200 dark:border-slate-800 shadow-2xl space-y-4">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">Shipping & Mock Checkout</h3>
            <p className="text-xs text-slate-500">Confirm delivery address for your SmartCart AI order.</p>

            <form onSubmit={handleCheckout} className="space-y-3 text-xs">
              <div>
                <label className="block font-semibold mb-1">Full Name</label>
                <input
                  type="text"
                  required
                  value={shippingAddress.name}
                  onChange={(e) => setShippingAddress({ ...shippingAddress, name: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 rounded-xl"
                />
              </div>
              <div>
                <label className="block font-semibold mb-1">Street Address</label>
                <input
                  type="text"
                  required
                  value={shippingAddress.street}
                  onChange={(e) => setShippingAddress({ ...shippingAddress, street: e.target.value })}
                  className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 rounded-xl"
                />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="block font-semibold mb-1">City</label>
                  <input
                    type="text"
                    required
                    value={shippingAddress.city}
                    onChange={(e) => setShippingAddress({ ...shippingAddress, city: e.target.value })}
                    className="w-full px-2.5 py-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 rounded-xl"
                  />
                </div>
                <div>
                  <label className="block font-semibold mb-1">State</label>
                  <input
                    type="text"
                    required
                    value={shippingAddress.state}
                    onChange={(e) => setShippingAddress({ ...shippingAddress, state: e.target.value })}
                    className="w-full px-2.5 py-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 rounded-xl"
                  />
                </div>
                <div>
                  <label className="block font-semibold mb-1">PIN Code</label>
                  <input
                    type="text"
                    required
                    value={shippingAddress.pin}
                    onChange={(e) => setShippingAddress({ ...shippingAddress, pin: e.target.value })}
                    className="w-full px-2.5 py-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 rounded-xl"
                  />
                </div>
              </div>

              <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-xl font-bold flex justify-between text-xs">
                <span>Total to Pay:</span>
                <span>₹{grandTotal.toLocaleString('en-IN')} (UPI / Mock)</span>
              </div>

              <div className="flex gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddressModal(false)}
                  className="flex-1 py-2.5 bg-slate-100 text-slate-700 rounded-xl font-semibold"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={checkingOut}
                  className="flex-1 py-2.5 bg-indigo-600 text-white rounded-xl font-bold hover:bg-indigo-500"
                >
                  {checkingOut ? 'Placing Order...' : 'Confirm Order'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}


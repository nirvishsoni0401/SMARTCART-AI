"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { Trash2, Plus, Minus, ShoppingBag, ShieldCheck, CheckCircle2, ArrowRight } from "lucide-react";
import { smartCartApi, type CartItem, type CartResponse } from "@/lib/smartcart-api";
import { Button } from "@/ui/components/ui/button";

interface CartViewClientProps {
	initialCart: CartResponse;
	locale: string;
	channel: string;
	defaultEmail?: string;
}

export function CartViewClient({ initialCart, locale, channel, defaultEmail }: CartViewClientProps) {
	const [cart, setCart] = useState<CartResponse>(initialCart);
	const [isLoading, setIsLoading] = useState(false);
	const [isCheckingOut, setIsCheckingOut] = useState(false);
	const [orderSuccess, setOrderSuccess] = useState<any | null>(null);
	const [error, setError] = useState<string | null>(null);

	// Checkout form fields
	const [fullName, setFullName] = useState("");
	const [email, setEmail] = useState(defaultEmail || "");
	const [address, setAddress] = useState("123 Innovation Way");
	const [city, setCity] = useState("San Jose");
	const [state, setState] = useState("CA");
	const [postalCode, setPostalCode] = useState("95134");

	const refreshCart = async () => {
		try {
			const res = await smartCartApi.getCart();
			setCart(res);
		} catch (e: any) {
			console.error("refreshCart error:", e);
		}
	};

	const handleUpdateQuantity = async (productId: number, newQty: number) => {
		setIsLoading(true);
		try {
			await smartCartApi.updateCart(productId, newQty);
			await refreshCart();
		} catch (e: any) {
			setError(e.message || "Failed to update item quantity");
		} finally {
			setIsLoading(false);
		}
	};

	const handleRemoveItem = async (productId: number) => {
		setIsLoading(true);
		try {
			await smartCartApi.removeFromCart(productId);
			await smartCartApi.trackEvent({ event_type: "remove_from_cart", product_id: productId });
			await refreshCart();
		} catch (e: any) {
			setError(e.message || "Failed to remove item");
		} finally {
			setIsLoading(false);
		}
	};

	const handlePlaceOrder = async (e: React.FormEvent) => {
		e.preventDefault();
		setError(null);
		setIsLoading(true);

		try {
			const orderRes = await smartCartApi.createOrder({
				shipping_address: {
					full_name: fullName.trim() || "Shopper",
					street_address: address,
					city,
					state,
					postal_code: postalCode,
					country: "United States",
				},
				customer_email: email.trim() || undefined,
				customer_name: fullName.trim() || undefined,
				payment_method: "safe_mock_card",
			});

			setOrderSuccess(orderRes.order);
			setCart({ items: [], total_items: 0, subtotal: 0, currency: "USD" });
		} catch (e: any) {
			setError(e.message || "Order placement failed. Please verify stock.");
		} finally {
			setIsLoading(false);
		}
	};

	if (orderSuccess) {
		return (
			<div className="mx-auto max-w-2xl py-12 text-center">
				<div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-emerald-100 text-emerald-600">
					<CheckCircle2 className="h-10 w-10" />
				</div>
				<h1 className="text-3xl font-extrabold tracking-tight text-foreground">Order Confirmed!</h1>
				<p className="mt-2 text-muted-foreground">
					Thank you for your purchase. We have received your order.
				</p>
				<div className="my-6 rounded-xl border bg-muted/30 p-6 text-left">
					<div className="flex justify-between border-b pb-3 text-sm">
						<span className="font-semibold text-foreground">Order ID:</span>
						<span className="font-mono text-primary">{orderSuccess.order_id}</span>
					</div>
					<div className="flex justify-between border-b py-3 text-sm">
						<span className="font-semibold text-foreground">Total Paid:</span>
						<span className="font-semibold">${orderSuccess.total_amount?.toFixed(2)}</span>
					</div>
					<div className="flex justify-between border-b py-3 text-sm">
						<span className="font-semibold text-foreground">Shipping To:</span>
						<span>{orderSuccess.shipping_address?.street_address}, {orderSuccess.shipping_address?.city}</span>
					</div>
					<div className="pt-3 text-sm">
						<span className="font-semibold text-foreground">Items ({orderSuccess.items?.length}):</span>
						<ul className="mt-2 space-y-1">
							{orderSuccess.items?.map((it: any) => (
								<li key={it.product_id} className="flex justify-between text-xs text-muted-foreground">
									<span>{it.quantity}x {it.name}</span>
									<span>${it.total_price?.toFixed(2)}</span>
								</li>
							))}
						</ul>
					</div>
				</div>
				<div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
					<Link
						href={`/${locale}/${channel}/products`}
						className="inline-flex items-center justify-center rounded-lg bg-primary px-6 py-3 font-semibold text-primary-foreground transition hover:opacity-90"
					>
						Continue Shopping <ArrowRight className="ml-2 h-4 w-4" />
					</Link>
					<Link
						href={`/${locale}/${channel}/orders`}
						className="inline-flex items-center justify-center rounded-lg border bg-background px-6 py-3 font-semibold text-foreground transition hover:bg-muted"
					>
						View Order History
					</Link>
				</div>
			</div>
		);
	}

	if (!cart.items || cart.items.length === 0) {
		return (
			<div className="py-16 text-center">
				<div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-muted text-muted-foreground">
					<ShoppingBag className="h-10 w-10" />
				</div>
				<h1 className="text-3xl font-extrabold tracking-tight text-foreground">Your Cart is Empty</h1>
				<p className="mx-auto mt-3 max-w-md text-sm text-muted-foreground">
					Explore thousands of top-rated multi-category products and personalized recommendations.
				</p>
				<div className="mt-8">
					<Link
						href={`/${locale}/${channel}/products`}
						className="inline-flex items-center rounded-lg bg-primary px-8 py-3.5 font-semibold text-primary-foreground shadow transition hover:opacity-90"
					>
						Start Browsing
					</Link>
				</div>
			</div>
		);
	}

	const subtotal = cart.subtotal || 0;
	const shippingThreshold = 75;
	const isFreeShipping = subtotal >= shippingThreshold;
	const shippingFee = isFreeShipping ? 0 : 9.99;
	const estimatedTax = Number((subtotal * 0.08).toFixed(2));
	const grandTotal = Number((subtotal + shippingFee + estimatedTax).toFixed(2));

	return (
		<div>
			<div className="mb-8 flex items-baseline justify-between border-b pb-4">
				<h1 className="text-3xl font-extrabold tracking-tight text-foreground">Shopping Cart</h1>
				<span className="text-sm text-muted-foreground">{cart.total_items} item(s)</span>
			</div>

			{error && (
				<div className="mb-6 rounded-lg bg-red-50 p-4 text-sm font-medium text-red-700">
					{error}
				</div>
			)}

			<div className="grid grid-cols-1 gap-12 lg:grid-cols-12">
				{/* Cart Items List */}
				<div className="lg:col-span-7">
					<div className="divide-y rounded-xl border bg-card">
						{cart.items.map((item: CartItem) => (
							<div key={item.product_id} className="flex gap-4 p-5">
								<div className="relative h-24 w-24 shrink-0 overflow-hidden rounded-lg border bg-muted">
									{item.image_url ? (
										<Image
											src={item.image_url}
											alt={item.name}
											fill
											className="object-contain p-2"
										/>
									) : (
										<div className="flex h-full w-full items-center justify-center text-xs text-muted-foreground">
											No image
										</div>
									)}
								</div>

								<div className="flex flex-1 flex-col justify-between">
									<div className="flex justify-between gap-2">
										<div>
											<span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
												{item.category}
											</span>
											<h3 className="text-base font-bold text-foreground">
												<Link
													href={`/${locale}/${channel}/products/${item.product_id}`}
													className="hover:underline"
												>
													{item.name}
												</Link>
											</h3>
											<p className="mt-0.5 text-xs text-muted-foreground">Brand: {item.brand}</p>
										</div>
										<div className="text-right">
											<span className="text-base font-bold text-foreground">
												${item.total_price.toFixed(2)}
											</span>
											<p className="text-xs text-muted-foreground">${item.price.toFixed(2)} each</p>
										</div>
									</div>

									<div className="mt-4 flex items-center justify-between">
										<div className="flex items-center rounded-lg border bg-muted/20">
											<button
												onClick={() => handleUpdateQuantity(item.product_id, item.quantity - 1)}
												disabled={isLoading || item.quantity <= 1}
												className="p-1.5 text-muted-foreground hover:text-foreground disabled:opacity-40"
												aria-label="Decrease quantity"
											>
												<Minus className="h-4 w-4" />
											</button>
											<span className="w-8 text-center text-sm font-semibold text-foreground">
												{item.quantity}
											</span>
											<button
												onClick={() => handleUpdateQuantity(item.product_id, item.quantity + 1)}
												disabled={isLoading || item.quantity >= item.stock}
												className="p-1.5 text-muted-foreground hover:text-foreground disabled:opacity-40"
												aria-label="Increase quantity"
											>
												<Plus className="h-4 w-4" />
											</button>
										</div>

										<button
											onClick={() => handleRemoveItem(item.product_id)}
											disabled={isLoading}
											className="flex items-center text-xs font-medium text-red-600 hover:text-red-700 hover:underline"
										>
											<Trash2 className="mr-1 h-3.5 w-3.5" /> Remove
										</button>
									</div>
								</div>
							</div>
						))}
					</div>
				</div>

				{/* Order Summary & Mock Checkout */}
				<div className="lg:col-span-5">
					<div className="rounded-2xl border bg-card p-6 shadow-sm">
						<h2 className="text-lg font-bold text-foreground">Order Summary</h2>

						<div className="mt-4 space-y-2 border-b pb-4 text-sm">
							<div className="flex justify-between text-muted-foreground">
								<span>Subtotal</span>
								<span className="font-semibold text-foreground">${subtotal.toFixed(2)}</span>
							</div>
							<div className="flex justify-between text-muted-foreground">
								<span>Shipping</span>
								<span className="font-semibold text-foreground">
									{isFreeShipping ? <span className="text-emerald-600">FREE</span> : `$${shippingFee.toFixed(2)}`}
								</span>
							</div>
							<div className="flex justify-between text-muted-foreground">
								<span>Estimated Tax (8%)</span>
								<span className="font-semibold text-foreground">${estimatedTax.toFixed(2)}</span>
							</div>
						</div>

						<div className="mt-4 flex justify-between text-base font-extrabold text-foreground">
							<span>Total Amount</span>
							<span className="text-primary">${grandTotal.toFixed(2)}</span>
						</div>

						{!isFreeShipping && (
							<div className="mt-4 rounded-lg bg-blue-50 p-3 text-xs text-blue-700">
								Add ${(shippingThreshold - subtotal).toFixed(2)} more to qualify for <strong>FREE Shipping</strong>!
							</div>
						)}

						{!isCheckingOut ? (
							<Button
								size="lg"
								onClick={() => setIsCheckingOut(true)}
								className="mt-6 w-full py-4 text-base font-bold shadow"
							>
								Proceed to Checkout
							</Button>
						) : (
							<form onSubmit={handlePlaceOrder} className="mt-6 space-y-4 border-t pt-4">
								<h3 className="text-sm font-bold text-foreground">Shipping & Mock Checkout</h3>
								<div>
									<label className="text-xs font-medium text-muted-foreground">Full Name</label>
									<input
										type="text"
										required
										value={fullName}
										onChange={(e) => setFullName(e.target.value)}
										placeholder="e.g. Jane Doe"
										className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
									/>
								</div>
								<div>
									<label className="text-xs font-medium text-muted-foreground">Email for Receipt</label>
									<input
										type="email"
										required
										value={email}
										onChange={(e) => setEmail(e.target.value)}
										placeholder="user@example.com"
										className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
									/>
								</div>
								<div>
									<label className="text-xs font-medium text-muted-foreground">Street Address</label>
									<input
										type="text"
										required
										value={address}
										onChange={(e) => setAddress(e.target.value)}
										className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
									/>
								</div>
								<div className="grid grid-cols-3 gap-2">
									<div>
										<label className="text-xs font-medium text-muted-foreground">City</label>
										<input
											type="text"
											required
											value={city}
											onChange={(e) => setCity(e.target.value)}
											className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
										/>
									</div>
									<div>
										<label className="text-xs font-medium text-muted-foreground">State</label>
										<input
											type="text"
											required
											value={state}
											onChange={(e) => setState(e.target.value)}
											className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
										/>
									</div>
									<div>
										<label className="text-xs font-medium text-muted-foreground">ZIP</label>
										<input
											type="text"
											required
											value={postalCode}
											onChange={(e) => setPostalCode(e.target.value)}
											className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
										/>
									</div>
								</div>

								<div className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground">
									<div className="flex items-center gap-1 font-semibold text-foreground">
										<ShieldCheck className="h-4 w-4 text-emerald-600" /> Safe Mock Payment
									</div>
									<p className="mt-1">
										No real credit card required. Stock is validated and deducted atomically in MongoDB.
									</p>
								</div>

								<Button
									type="submit"
									size="lg"
									disabled={isLoading}
									className="w-full py-4 text-base font-bold shadow"
								>
									{isLoading ? "Processing Order..." : `Confirm & Place Order ($${grandTotal.toFixed(2)})`}
								</Button>
							</form>
						)}
					</div>
				</div>
			</div>
		</div>
	);
}


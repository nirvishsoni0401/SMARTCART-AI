import { Suspense } from "react";
import { type Metadata } from "next";
import { cookies } from "next/headers";
import { API_BASE_URL } from "@/lib/smartcart-api";
import { CartViewClient } from "./cart-view-client";

export const metadata: Metadata = {
	title: "Shopping Cart | SmartCart AI",
	description: "Review items in your cart and complete your order with SmartCart AI.",
};

export default function CartPage(props: { params: Promise<{ locale: string; channel: string }> }) {
	return (
		<section className="container-content py-8">
			<Suspense fallback={<CartSkeleton />}>
				<CartLoader params={props.params} />
			</Suspense>
		</section>
	);
}

async function CartLoader({ params: paramsPromise }: { params: Promise<{ locale: string; channel: string }> }) {
	const params = await paramsPromise;
	const cookieStore = await cookies();
	const token = cookieStore.get("access_token")?.value;
	const sessionId = cookieStore.get("smartcart_session_id")?.value || "guest_default_session";
	const userEmail = cookieStore.get("smartcart_user_email")?.value;

	const headers: Record<string, string> = { "x-session-id": sessionId };
	if (token) headers["Authorization"] = `Bearer ${token}`;

	let cartData = { items: [], total_items: 0, subtotal: 0, currency: "USD" };

	try {
		const res = await fetch(`${API_BASE_URL}/cart/`, { headers, cache: "no-store" });
		if (res.ok) {
			cartData = await res.json();
		}
	} catch (e) {
		console.error("CartLoader error:", e);
	}

	return (
		<CartViewClient
			initialCart={cartData}
			locale={params.locale}
			channel={params.channel}
			defaultEmail={userEmail}
		/>
	);
}

function CartSkeleton() {
	return (
		<div className="animate-pulse space-y-4 py-8">
			<div className="h-8 w-48 rounded bg-muted" />
			<div className="h-64 rounded bg-muted" />
		</div>
	);
}

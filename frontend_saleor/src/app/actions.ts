"use server";

import { cookies } from "next/headers";
import { refresh } from "next/cache";
import { API_BASE_URL } from "@/lib/smartcart-api";

export async function logout() {
	const cookieStore = await cookies();
	cookieStore.delete("access_token");
	cookieStore.delete("smartcart_user_email");
}

export async function saveCheckoutId(channel: string, checkoutId: string) {
	// No-op for mock checkout
}

export async function clearCheckout(channel: string) {
	const cookieStore = await cookies();
	const token = cookieStore.get("access_token")?.value;
	const sessionId = cookieStore.get("smartcart_session_id")?.value || "guest_default_session";
	const headers: Record<string, string> = { "x-session-id": sessionId };
	if (token) headers["Authorization"] = `Bearer ${token}`;

	try {
		await fetch(`${API_BASE_URL}/cart/`, { method: "DELETE", headers });
	} catch (e) {
		console.error("clearCheckout error:", e);
	}
	refresh();
}

export async function deleteCartLine(checkoutId: string, lineId: string) {
	const cookieStore = await cookies();
	const token = cookieStore.get("access_token")?.value;
	const sessionId = cookieStore.get("smartcart_session_id")?.value || "guest_default_session";
	const headers: Record<string, string> = { "x-session-id": sessionId };
	if (token) headers["Authorization"] = `Bearer ${token}`;

	const prodId = parseInt(String(lineId).replace(/\D/g, ""), 10);
	if (!isNaN(prodId)) {
		try {
			await fetch(`${API_BASE_URL}/cart/${prodId}`, { method: "DELETE", headers });
			// Also track remove_from_cart event for bandit
			await fetch(`${API_BASE_URL}/events/`, {
				method: "POST",
				headers: { ...headers, "Content-Type": "application/json" },
				body: JSON.stringify({ event_type: "remove_from_cart", product_id: prodId }),
			});
		} catch (e) {
			console.error("deleteCartLine error:", e);
		}
	}

	refresh();
}

export async function updateCartLineQuantity(checkoutId: string, lineId: string, quantity: number) {
	if (quantity < 1) {
		return deleteCartLine(checkoutId, lineId);
	}

	const cookieStore = await cookies();
	const token = cookieStore.get("access_token")?.value;
	const sessionId = cookieStore.get("smartcart_session_id")?.value || "guest_default_session";
	const headers: Record<string, string> = { "x-session-id": sessionId, "Content-Type": "application/json" };
	if (token) headers["Authorization"] = `Bearer ${token}`;

	const prodId = parseInt(String(lineId).replace(/\D/g, ""), 10);
	if (!isNaN(prodId)) {
		try {
			await fetch(`${API_BASE_URL}/cart/update`, {
				method: "PUT",
				headers,
				body: JSON.stringify({ product_id: prodId, quantity }),
			});
		} catch (e) {
			console.error("updateCartLineQuantity error:", e);
		}
	}

	refresh();
}

export async function addVariantToCart(input: {
	channel: string;
	localeSlug: string;
	variantId: string;
	value?: number;
	currency?: string;
}) {
	const cookieStore = await cookies();
	const token = cookieStore.get("access_token")?.value;
	const sessionId = cookieStore.get("smartcart_session_id")?.value || "guest_default_session";
	const headers: Record<string, string> = { "x-session-id": sessionId, "Content-Type": "application/json" };
	if (token) headers["Authorization"] = `Bearer ${token}`;

	const prodId = parseInt(String(input.variantId).replace(/\D/g, ""), 10);
	if (!isNaN(prodId)) {
		try {
			await fetch(`${API_BASE_URL}/cart/add`, {
				method: "POST",
				headers,
				body: JSON.stringify({ product_id: prodId, quantity: 1 }),
			});
			// Log add_to_cart event for LinUCB bandit
			await fetch(`${API_BASE_URL}/events/`, {
				method: "POST",
				headers,
				body: JSON.stringify({
					event_type: "add_to_cart",
					product_id: prodId,
					metadata: { value: input.value, currency: input.currency },
				}),
			});
		} catch (e) {
			console.error("addVariantToCart error:", e);
		}
	}

	refresh();
}

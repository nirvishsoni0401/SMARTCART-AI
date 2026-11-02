import "server-only";

import { cookies } from "next/headers";
import { API_BASE_URL } from "@/lib/smartcart-api";

export async function getCurrentUserOrders(localeSlug: string, first: number, after?: string | null) {
	try {
		const cookieStore = await cookies();
		const token = cookieStore.get("access_token")?.value;
		const email = cookieStore.get("smartcart_user_email")?.value;

		const headers: Record<string, string> = {};
		if (token) headers["Authorization"] = `Bearer ${token}`;

		const qs = email ? `?email=${encodeURIComponent(email)}` : "";
		const res = await fetch(`${API_BASE_URL}/orders/${qs}`, {
			headers,
			cache: "no-store",
		});

		if (!res.ok) {
			return { ok: false, error: { message: "Failed to fetch orders" } };
		}

		const data = await res.json();
		const orders = data.orders || [];

		const edges = orders.map((ord: any) => ({
			node: {
				id: ord.order_id,
				number: ord.order_id,
				created: ord.created_at,
				status: (ord.order_status || "CONFIRMED").toUpperCase(),
				total: {
					gross: {
						amount: ord.total_amount,
						currency: ord.currency || "USD",
					},
				},
				lines: (ord.items || []).map((it: any) => ({
					id: `line-${it.product_id}`,
					productName: it.name,
					quantity: it.quantity,
					thumbnail: { url: it.image_url, alt: it.name },
				})),
			},
		}));

		return {
			ok: true,
			data: {
				me: {
					orders: {
						totalCount: orders.length,
						edges,
						pageInfo: {
							hasNextPage: false,
							hasPreviousPage: false,
						},
					},
				},
			},
		};
	} catch (err: any) {
		console.error("getCurrentUserOrders error:", err);
		return { ok: false, error: { message: err.message } };
	}
}

export async function getOrdersForNumberLookup(localeSlug: string, first = 100) {
	return getCurrentUserOrders(localeSlug, first);
}

import { cookies } from "next/headers";
import { API_BASE_URL } from "@/lib/smartcart-api";
import { CartButton } from "@/ui/components/cart";

export const CartNavItem = async ({ channel, localeSlug }: { channel: string; localeSlug: string }) => {
	let lineCount = 0;
	try {
		const cookieStore = await cookies();
		const token = cookieStore.get("access_token")?.value;
		const sessionId = cookieStore.get("smartcart_session_id")?.value || "guest_default_session";

		const headers: Record<string, string> = { "x-session-id": sessionId };
		if (token) headers["Authorization"] = `Bearer ${token}`;

		const res = await fetch(`${API_BASE_URL}/cart/`, {
			headers,
			cache: "no-store",
		});

		if (res.ok) {
			const data = await res.json();
			lineCount = data.total_items || 0;
		}
	} catch {
		lineCount = 0;
	}

	return <CartButton itemCount={lineCount} />;
};

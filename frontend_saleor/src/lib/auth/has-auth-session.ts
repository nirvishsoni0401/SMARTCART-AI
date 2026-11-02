import "server-only";

import { cookies } from "next/headers";

export async function hasAuthSession(): Promise<boolean> {
	try {
		const cookieStore = await cookies();
		return cookieStore.has("access_token") || cookieStore.has("smartcart_user_email");
	} catch {
		return false;
	}
}

export async function getAuthTokenPresence(): Promise<{ hasAccess: boolean; hasRefresh: boolean }> {
	const has = await hasAuthSession();
	return { hasAccess: has, hasRefresh: has };
}

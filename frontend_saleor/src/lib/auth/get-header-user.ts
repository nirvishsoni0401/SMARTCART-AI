import "server-only";

import { cache } from "react";
import { cookies } from "next/headers";
import { API_BASE_URL } from "@/lib/smartcart-api";

export type HeaderUser = {
	id: string;
	email: string;
	firstName?: string;
	lastName?: string;
};

export type HeaderAuthState =
	| { status: "authenticated"; user: HeaderUser }
	| { status: "guest" }
	| { status: "unavailable" };

export const getHeaderAuthState = cache(async (): Promise<HeaderAuthState> => {
	try {
		const cookieStore = await cookies();
		const token = cookieStore.get("access_token")?.value;
		if (!token) {
			return { status: "guest" };
		}

		const res = await fetch(`${API_BASE_URL}/auth/me`, {
			headers: {
				Authorization: `Bearer ${token}`,
			},
			cache: "no-store",
		});

		if (!res.ok) {
			return { status: "guest" };
		}

		const data = await res.json();
		const user = data.user;

		return {
			status: "authenticated",
			user: {
				id: user.id,
				email: user.email,
				firstName: user.name || user.email.split("@")[0],
			},
		};
	} catch (err) {
		return { status: "guest" };
	}
});

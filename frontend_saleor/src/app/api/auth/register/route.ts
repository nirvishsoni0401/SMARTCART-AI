import { NextRequest, NextResponse } from "next/server";
import { API_BASE_URL } from "@/lib/smartcart-api";

interface RegisterRequest {
	email: string;
	password: string;
	firstName?: string;
	lastName?: string;
	channel?: string;
}

export async function POST(request: NextRequest) {
	let body: RegisterRequest;
	try {
		body = (await request.json()) as RegisterRequest;
	} catch {
		return NextResponse.json(
			{ errors: [{ message: "Invalid request body", code: "INVALID_JSON" }] },
			{ status: 400 },
		);
	}

	const { email, password, firstName, lastName } = body;

	if (!email || !password) {
		return NextResponse.json(
			{ errors: [{ message: "Email and password are required", code: "REQUIRED" }] },
			{ status: 400 },
		);
	}

	const fullName = [firstName, lastName].filter(Boolean).join(" ") || "SmartCart Shopper";

	try {
		const res = await fetch(`${API_BASE_URL}/auth/register`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				name: fullName,
				email,
				password,
			}),
		});

		const data = await res.json();

		if (!res.ok) {
			return NextResponse.json(
				{ errors: [{ message: data.detail || "Registration failed", code: "REGISTRATION_FAILED" }] },
				{ status: res.status }
			);
		}

		const response = NextResponse.json({ ok: true, user: data.user, access_token: data.access_token });
		
		if (data.access_token) {
			response.cookies.set("access_token", data.access_token, {
				httpOnly: true,
				sameSite: "lax",
				path: "/",
				maxAge: 7 * 24 * 3600,
			});
			response.cookies.set("smartcart_user_email", data.user?.email || "", {
				path: "/",
				maxAge: 7 * 24 * 3600,
			});
		}

		return response;
	} catch (err: any) {
		return NextResponse.json(
			{ errors: [{ message: err.message || "Failed to reach backend", code: "NETWORK_ERROR" }] },
			{ status: 500 }
		);
	}
}

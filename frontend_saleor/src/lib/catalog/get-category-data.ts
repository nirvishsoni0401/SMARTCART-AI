import "server-only";

import { smartCartApi } from "@/lib/smartcart-api";

export async function getCategoryData(slug: string, channel: string, localeSlug: string) {
	try {
		const decodedSlug = decodeURIComponent(slug).toLowerCase();
		const name = decodedSlug.charAt(0).toUpperCase() + decodedSlug.slice(1).replace(/-/g, " ");

		return {
			id: decodedSlug,
			name,
			slug: decodedSlug,
			description: `Browse all products in ${name} on SmartCart AI.`,
			seoTitle: `${name} | SmartCart AI`,
			seoDescription: `Find top products, competitive prices and adaptive recommendations for ${name}.`,
			backgroundImage: {
				url: "/placeholder.svg",
				alt: name,
			},
			translation: null
		} as any;
	} catch (err) {
		console.error(`[getCategoryData] Error for category ${slug}:`, err);
		return null;
	}
}

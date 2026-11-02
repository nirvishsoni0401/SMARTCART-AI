import "server-only";

import { smartCartApi, type SmartCartProduct } from "@/lib/smartcart-api";

export async function getFeaturedProducts(
	channel: string,
	localeSlug: string,
	limit = 12,
	collectionSlug = "featured-products",
) {
	try {
		const res = await smartCartApi.getProducts({
			limit,
			sort_by: "popularity"
		});

		const items = res.products || [];

		return items.map((p: SmartCartProduct) => {
			const price = Number(p.discounted_price || p.price) || 0;
			const id = String(p.product_id || p.external_id);
			const title = p.title || p.name;
			const imgUrl = p.thumbnail || p.image_url || p.images?.[0] || "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=600&q=80";

			return {
				id,
				name: title,
				slug: id,
				created: "2026-01-01T00:00:00.000Z",
				thumbnail: {
					url: imgUrl,
					alt: title
				},
				thumbnail256: {
					url: imgUrl,
					alt: title
				},
				thumbnail512: {
					url: imgUrl,
					alt: title
				},
				category: {
					id: p.category,
					name: p.category,
					slug: p.category_slug || p.category.toLowerCase().replace(/ & /g, "-and-").replace(/ /g, "-"),
					translation: null,
					backgroundImage: {
						url: imgUrl,
						alt: p.category
					},
					backgroundImage256: {
						url: imgUrl,
						alt: p.category
					},
					backgroundImage512: {
						url: imgUrl,
						alt: p.category
					}
				},
				pricing: {
					priceRange: {
						start: {
							gross: {
								amount: price,
								currency: "INR"
							}
						},
						stop: {
							gross: {
								amount: price,
								currency: "INR"
							}
						}
					},
					priceRangeUndiscounted: null
				},
				productVariants: {
					totalCount: 1,
					edges: [
						{
							node: {
								id: `var-${id}`,
								name: "Standard",
								selectionAttributes: []
							}
						}
					]
				}
			} as any;
		});
	} catch (err) {
		console.error("[getFeaturedProducts] Failed to fetch from FastAPI:", err);
		return [];
	}
}

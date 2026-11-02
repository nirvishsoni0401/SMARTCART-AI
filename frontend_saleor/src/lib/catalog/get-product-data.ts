import "server-only";

import { smartCartApi, type SmartCartProduct } from "@/lib/smartcart-api";
import type { ProductShell, ProductVariantsForPdpResult, PdpVariant } from "./get-product-data.types";

export type { ProductShell, ProductVariantsForPdpResult, PdpVariant };

/**
 * Fetch and construct ProductShell from FastAPI backend
 */
export async function getProductData(
	slug: string,
	channel: string,
	localeSlug: string,
): Promise<ProductShell | null> {
	try {
		const decodedSlug = decodeURIComponent(slug);
		const p = await smartCartApi.getProductById(decodedSlug);
		if (!p) return null;

		const currentPrice = Number(p.discounted_price || p.price) || 0;
		const originalPrice = Number(p.price) || currentPrice;
		const title = p.title || p.name;
		const id = String(p.product_id || p.external_id);

		const imgUrl = p.thumbnail || p.image_url || p.images?.[0] || "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=800&q=80";
		const rawImages = p.images && p.images.length > 0 ? p.images : [imgUrl];
		const mediaList = rawImages.map((url: string, i: number) => ({
			id: `media-${i}`,
			url,
			url256: url,
			url512: url,
			url1024: url,
			alt: `${title} - Image ${i + 1}`,
			type: "IMAGE"
		}));

		// Build assigned attributes from specifications
		const assignedAttrs: any[] = [
			{ attribute: { name: "Brand", slug: "brand" }, values: [{ name: p.brand || "SmartCart Verified" }] },
			{ attribute: { name: "Category", slug: "category" }, values: [{ name: p.category }] },
			...(p.subcategory ? [{ attribute: { name: "Subcategory", slug: "subcategory" }, values: [{ name: p.subcategory }] }] : []),
			{ attribute: { name: "Customer Rating", slug: "rating" }, values: [{ name: `${p.rating?.toFixed(1) || "4.5"} / 5.0 ★ (${(p.review_count || 120).toLocaleString()} reviews)` }] },
			{ attribute: { name: "Stock Availability", slug: "availability" }, values: [{ name: `${p.availability || "In Stock"} (${p.stock_quantity || p.stock || 50} units)` }] },
			...(p.seller ? [{ attribute: { name: "Verified Seller", slug: "seller" }, values: [{ name: p.seller }] }] : []),
		];

		if (p.specifications && typeof p.specifications === "object") {
			for (const [k, v] of Object.entries(p.specifications)) {
				if (k !== "Brand" && k !== "Color" && v) {
					assignedAttrs.push({
						attribute: { name: k, slug: k.toLowerCase().replace(/\s+/g, "-") },
						values: [{ name: String(v) }]
					});
				}
			}
		}

		const shell: any = {
			id,
			name: title,
			slug: id,
			description: p.description,
			seoTitle: `${title} | SmartCart AI India`,
			seoDescription: p.description,
			isAvailable: (p.stock_quantity || p.stock || 0) > 0,
			category: {
				id: p.category,
				name: p.category,
				slug: p.category_slug || p.category.toLowerCase().replace(/ & /g, "-and-").replace(/ /g, "-"),
				translation: null
			},
			thumbnail: { url: imgUrl, alt: title },
			thumbnail256: { url: imgUrl, alt: title },
			thumbnail512: { url: imgUrl, alt: title },
			media: mediaList,
			pricing: {
				priceRange: {
					start: { gross: { amount: currentPrice, currency: "INR" } },
					stop: { gross: { amount: currentPrice, currency: "INR" } }
				},
				priceRangeUndiscounted: originalPrice > currentPrice ? {
					start: { gross: { amount: originalPrice, currency: "INR" } },
					stop: { gross: { amount: originalPrice, currency: "INR" } }
				} : null
			},
			productVariants: {
				totalCount: 1,
				edges: [
					{
						node: {
							id: `var-${id}`,
							name: "Standard Edition",
							sku: `SKU-${id}`,
							quantityAvailable: p.stock_quantity || p.stock || 50,
							pricing: {
								price: { gross: { amount: currentPrice, currency: "INR" } }
							},
							media: mediaList,
							selectionAttributes: []
						}
					}
				]
			},
			assignedAttributes: assignedAttrs
		};

		return shell as ProductShell;
	} catch (err) {
		console.error(`[getProductData] Error fetching product ${slug}:`, err);
		return null;
	}
}

export async function resolvePdpVariants(
	product: ProductShell,
	channel: string,
	localeSlug: string,
	options?: { variantId?: string | null; sku?: string | null },
): Promise<ProductVariantsForPdpResult> {
	const rawPrice = product.pricing?.priceRange?.start?.gross;
	const media = product.media || [];
	const isAvail = (product as any).isAvailable ?? true;

	const variantNode: any = {
		id: `var-${product.id}`,
		name: "Standard Edition",
		sku: `SKU-${product.id}`,
		quantityAvailable: isAvail ? 50 : 0,
		pricing: {
			price: {
				gross: {
					amount: rawPrice?.amount ?? 0,
					currency: "INR"
				}
			}
		},
		media,
		selectionAttributes: []
	};

	return {
		variants: [variantNode],
		totalCount: 1,
		overBudget: false,
		strategy: "matrix"
	};
}

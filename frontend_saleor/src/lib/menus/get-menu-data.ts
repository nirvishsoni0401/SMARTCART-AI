import "server-only";

import { smartCartApi } from "@/lib/smartcart-api";

export type MenuItem = {
	id: string;
	name?: string | null;
	category?: { name?: string | null; slug?: string | null } | null;
	collection?: { name?: string | null; slug?: string | null } | null;
	page?: { title?: string | null; slug?: string | null } | null;
	url?: string | null;
	children?: MenuItem[];
};

export async function getNavbarMenuItems(channel: string, localeSlug: string): Promise<MenuItem[]> {
	try {
		const catsData = await smartCartApi.getCategories().catch(() => ({ categories: [], details: [] }));
		const categoryChildren: MenuItem[] = (catsData.details || []).map((cat) => ({
			id: `cat-${cat.slug}`,
			name: cat.name,
			url: `/categories/${cat.slug}`,
			category: {
				name: cat.name,
				slug: cat.slug,
			},
		}));

		return [
			{
				id: "nav-all-products",
				name: "All Products (10,000+)",
				url: "/products",
			},
			{
				id: "nav-deals",
				name: "Today's Deals",
				url: "/products?sort_by=discount_desc",
			},
			{
				id: "nav-categories",
				name: "Departments",
				url: "/categories/electronics",
				children: categoryChildren,
			},
			{
				id: "nav-recommended",
				name: "AI Recommendations",
				url: "/#recommended",
			},
			{
				id: "nav-analytics",
				name: "Bandit Dashboard",
				url: "/analytics",
			},
		];
	} catch (err) {
		return [
			{ id: "nav-products", name: "Products", url: "/products" },
			{ id: "nav-deals", name: "Deals", url: "/products?sort_by=discount_desc" },
			{ id: "nav-analytics", name: "Bandit Dashboard", url: "/analytics" },
		];
	}
}

export async function getFooterMenuItems(channel: string, localeSlug: string): Promise<MenuItem[]> {
	return [
		{ id: "foot-home", name: "Home", url: "/" },
		{ id: "foot-products", name: "All 10,000+ Products", url: "/products" },
		{ id: "foot-deals", name: "Super Deals", url: "/products?sort_by=discount_desc" },
		{ id: "foot-cart", name: "Shopping Cart", url: "/cart" },
		{ id: "foot-analytics", name: "Adaptive LinUCB Analytics", url: "/analytics" },
	];
}

/**
 * Brand Configuration
 *
 * Centralized branding settings for the storefront.
 * Update these values when customizing for a new store.
 *
 * @example
 * ```tsx
 * import { brandConfig } from "@/config/brand";
 *
 * <title>{brandConfig.siteName}</title>
 * <Suspense><CopyrightText /></Suspense> // never `new Date()` during render (sync IO under Cache Components)
 * ```
 */

export const brandConfig = {
	/** Site name used in titles, metadata, and headers */
	siteName: "SmartCart AI",

	/** Legal entity name for copyright notices */
	copyrightHolder: "SmartCart AI Inc.",

	/** Organization name for structured data (JSON-LD) */
	organizationName: "SmartCart AI",

	/** Default brand name for products without a brand */
	defaultBrand: "SmartCart AI",

	/** Tagline/description for the store */
	tagline: "Adaptive AI-Powered E-Commerce Platform",

	/** Homepage meta description */
	description: "Shop smarter with SmartCart AI. Multi-category e-commerce with real-time adaptive reinforcement learning recommendations.",

	/** Logo aria-label for accessibility */
	logoAriaLabel: "SmartCart AI",

	/** Title template - %s will be replaced with page title */
	titleTemplate: "%s | SmartCart AI",

	/** Social media handles */
	social: {
		/** Twitter/X handle (without @) - set to null to disable */
		twitter: "SmartCartAI" as string | null,
		/** Instagram handle (without @) - set to null to disable */
		instagram: "SmartCartAI" as string | null,
		/** Facebook page URL - set to null to disable */
		facebook: null as string | null,
	},
} as const;

/**
 * Helper to format page title using brand template.
 */
export function formatPageTitle(title: string): string {
	return brandConfig.titleTemplate.replace("%s", title);
}

/**
 * Get copyright text with specified year.
 * Render through `CopyrightText` inside `<Suspense>`: it calls `io()` before reading the clock.
 */
export function getCopyrightText(year: number = new Date().getFullYear()): string {
	return `© ${year} ${brandConfig.copyrightHolder}. All rights reserved.`;
}

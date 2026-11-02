import { fixtureListingProvider } from "@/lib/listing/providers/fixture";
import { saleorListingProvider } from "@/lib/listing/providers/saleor";
import { smartcartListingProvider } from "@/lib/listing/providers/smartcart";
import type { ListingProvider } from "@/lib/listing/provider";
import type { ListingSurfaceKind } from "@/lib/storefront/contract/listing";

export const LISTING_PROVIDER_REGISTRY = {
	smartcart: smartcartListingProvider,
	saleor: saleorListingProvider,
	fixture: fixtureListingProvider,
} as const satisfies Record<string, ListingProvider>;

export const LISTING_PROVIDERS = {
	all: "smartcart",
	category: "smartcart",
	collection: "smartcart",
	search: "smartcart",
} as const satisfies Record<ListingSurfaceKind, keyof typeof LISTING_PROVIDER_REGISTRY>;

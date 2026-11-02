export interface ProductShell {
	id: string;
	name: string;
	slug: string;
	description?: string | null;
	seoTitle?: string | null;
	seoDescription?: string | null;
	isAvailable?: boolean;
	category?: {
		id: string;
		name: string;
		slug: string;
		translation?: any;
	} | null;
	thumbnail?: {
		url: string;
		alt?: string | null;
	} | null;
	thumbnail256?: {
		url: string;
		alt?: string | null;
	} | null;
	thumbnail512?: {
		url: string;
		alt?: string | null;
	} | null;
	media?: Array<{
		id: string;
		url: string;
		alt?: string | null;
		type?: string;
	}>;
	pricing?: {
		priceRange?: {
			start?: { gross?: { amount: number; currency: string } };
			stop?: { gross?: { amount: number; currency: string } };
		};
		priceRangeUndiscounted?: any;
	} | null;
	productVariants?: {
		totalCount?: number;
		edges?: Array<{
			node: PdpVariant;
		}>;
	} | null;
	assignedAttributes?: Array<{
		attribute: { name: string; slug?: string | null };
		values?: Array<{ name: string; slug?: string | null }>;
	}>;
	productType?: {
		slug: string;
	};
}

export interface PdpVariant {
	id: string;
	name: string;
	sku?: string | null;
	quantityAvailable?: number | null;
	pricing?: {
		price?: { gross?: { amount: number; currency: string } };
	} | null;
	media?: Array<{
		id: string;
		url: string;
		alt?: string | null;
	}>;
	selectionAttributes?: Array<{
		attribute?: { slug?: string | null };
		values?: Array<{ name?: string | null; slug?: string | null }>;
	}>;
}

export interface ProductVariantsForPdpResult {
	variants: PdpVariant[];
	totalCount: number;
	overBudget: boolean;
	strategy: "matrix" | "over_budget" | "external";
}


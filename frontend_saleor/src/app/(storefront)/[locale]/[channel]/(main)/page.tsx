import { Suspense } from "react";
import { brandConfig } from "@/config/brand";
import { resolveLocaleFromSlug } from "@/config/locale";
import { getFeaturedProducts } from "@/lib/catalog/get-featured-products";
import { resolveChannelCurrency } from "@/lib/channels/resolve-channel-currency";
import { buildPolicyLabelValues } from "@/lib/content";
import { formatContentLabel } from "@/lib/content/format-label";
import { getStorefrontContent } from "@/lib/content/server";
import { buildSaleorSrcSet } from "@/lib/images";
import { PaperSignEditorialPlaceholder } from "@/ui/components/shared/paper-sign";
import { EditorialHero } from "@/ui/sections/editorial-hero/editorial-hero";
import { MediaHero } from "@/ui/sections/media-hero/media-hero";
import { MulticolumnSection } from "@/ui/sections/multicolumn-section/multicolumn-section";
import { RichTextBlock } from "@/ui/sections/rich-text-block/rich-text-block";
import { RecommendationSection } from "@/ui/components/recommendations/recommendation-section";
import { MarketplaceCarousel } from "@/ui/components/marketplace/marketplace-carousel";
import { CategoryShowcase } from "@/ui/components/marketplace/category-showcase";
import { PopularBrands } from "@/ui/components/marketplace/popular-brands";
import { DealsBanner } from "@/ui/components/marketplace/deals-banner";
import { smartCartApi } from "@/lib/smartcart-api";

export const metadata = {
	description: brandConfig.description,
};

type FeaturedProduct = Awaited<ReturnType<typeof getFeaturedProducts>>[number];
type HomeParams = Promise<{ locale: string; channel: string }>;

function productThumbnailSrcSet(product: FeaturedProduct) {
	return buildSaleorSrcSet([
		{ width: 256, url: product.thumbnail256?.url },
		{ width: 512, url: product.thumbnail512?.url },
		{ width: 1024, url: product.thumbnail?.url },
	]);
}

function pickImage(product: FeaturedProduct | undefined) {
	if (!product?.thumbnail?.url) return null;
	return {
		url: product.thumbnail.url,
		alt: product.thumbnail.alt || product.name || "",
		srcSet: productThumbnailSrcSet(product),
	};
}

function HomePageFallback() {
	return (
		<div aria-hidden="true">
			<div className="min-h-[calc(100svh-var(--chrome-offset))] bg-secondary">
				<div className="container-content flex h-full min-h-[calc(100svh-var(--chrome-offset))] flex-col justify-end pb-16 pt-24">
					<div className="h-3 w-24 animate-pulse rounded bg-muted" />
					<div className="mt-4 h-10 w-2/3 max-w-xl animate-pulse rounded bg-muted" />
					<div className="mt-3 h-4 w-1/2 max-w-md animate-pulse rounded bg-muted" />
					<div className="mt-8 h-11 w-36 animate-pulse rounded bg-muted" />
				</div>
			</div>
		</div>
	);
}

async function HomePageContent({ params }: { params: HomeParams }) {
	const { locale, channel } = await params;
	const content = await getStorefrontContent(channel, locale);
	const { hero, brandStory, values } = content.surfaces.homepage;

	// Source product imagery from catalog
	const products = await getFeaturedProducts(channel, locale, 4);
	const heroProduct = products[0];
	const heroImage = pickImage(heroProduct);

	// Fetch dynamic marketplace sections from FastAPI
	let categoriesDetails: any[] = [];
	let dealsProducts: any[] = [];
	let trendingProducts: any[] = [];
	let bestsellersProducts: any[] = [];
	let topRatedProducts: any[] = [];

	try {
		const [catsRes, dealsRes, trendRes, bestRes, topRes] = await Promise.all([
			smartCartApi.getCategories().catch(() => ({ categories: [], details: [] })),
			smartCartApi.getTodaysDeals(12).catch(() => ({ products: [] })),
			smartCartApi.getTrendingProducts(12).catch(() => ({ products: [] })),
			smartCartApi.getBestsellers(12).catch(() => ({ products: [] })),
			smartCartApi.getTopRated(12).catch(() => ({ products: [] })),
		]);
		categoriesDetails = catsRes.details || [];
		dealsProducts = dealsRes.products || [];
		trendingProducts = trendRes.products || [];
		bestsellersProducts = bestRes.products || [];
		topRatedProducts = topRes.products || [];
	} catch (e) {
		console.error("Error fetching homepage marketplace data:", e);
	}

	const currency = await resolveChannelCurrency(channel);
	const policyValues = buildPolicyLabelValues(content.policies, {
		currency,
		locale: resolveLocaleFromSlug(locale).bcp47,
	});
	const valueColumns = values.columns.map((column) => ({
		...column,
		text: formatContentLabel(column.text, policyValues),
	}));

	return (
		<>
			{/* 1. Hero Banner */}
			{hero.backgroundImage ? (
				<MediaHero
					id="homepage-hero-heading"
					eyebrow={hero.eyebrow}
					heading={hero.heading}
					subheading={hero.subheading}
					image={hero.backgroundImage}
					align="left"
					height="fold"
					primaryCta={{ label: hero.primaryCtaLabel, href: `/${locale}/${channel}/products` }}
				/>
			) : (
				<EditorialHero
					eyebrow="10,000+ Products • Adaptive AI Platform"
					heading="Next-Gen E-Commerce with Intelligent Recommendations"
					subheading="Discover top-rated Electronics, Fashion, Home & Kitchen, Books, Fitness and more with real-time LinUCB Bandit personalization."
					image={heroImage?.url}
					imageSrcSet={heroImage?.srcSet}
					imageAlt={heroImage?.alt ?? ""}
					primaryCta={{ label: "Browse 10,000+ Products", href: `/${locale}/${channel}/products` }}
					placeholder={<PaperSignEditorialPlaceholder />}
				/>
			)}

			{/* 2. Today's Super Deals Flash Banner */}
			<DealsBanner locale={locale} channel={channel} />

			{/* 3. Shop by Category (10 Major Categories) */}
			{categoriesDetails.length > 0 && (
				<CategoryShowcase
					categories={categoriesDetails}
					locale={locale}
					channel={channel}
				/>
			)}

			{/* 4. Today's Deals Section */}
			{dealsProducts.length > 0 && (
				<MarketplaceCarousel
					title="Today's Blockbuster Deals"
					subtitle="Handpicked mega discounts up to 65% OFF across top brands and categories."
					badgeText="Deal of the Day"
					badgeColor="rose"
					viewAllHref={`/${locale}/${channel}/products?sort_by=discount_desc`}
					products={dealsProducts}
					locale={locale}
					channel={channel}
				/>
			)}

			{/* 5. Adaptive AI Personalized Recommendations (LinUCB Bandits) */}
			<RecommendationSection locale={locale} channel={channel} />

			{/* 6. Trending Products */}
			{trendingProducts.length > 0 && (
				<MarketplaceCarousel
					title="Trending Products"
					subtitle="Highest velocity customer favorites and most searched products this week."
					badgeText="Trending Now"
					badgeColor="amber"
					viewAllHref={`/${locale}/${channel}/products?sort_by=popularity`}
					products={trendingProducts}
					locale={locale}
					channel={channel}
				/>
			)}

			{/* 7. Best Sellers */}
			{bestsellersProducts.length > 0 && (
				<MarketplaceCarousel
					title="Customer Best Sellers"
					subtitle="Highest rated products with thousands of verified customer reviews."
					badgeText="Best Sellers"
					badgeColor="emerald"
					viewAllHref={`/${locale}/${channel}/products?sort_by=rating_desc`}
					products={bestsellersProducts}
					locale={locale}
					channel={channel}
				/>
			)}

			{/* 8. Top Rated Products */}
			{topRatedProducts.length > 0 && (
				<MarketplaceCarousel
					title="Top Rated (4.5★ & Above)"
					subtitle="Consistently highest satisfaction ratings from verified buyers."
					badgeText="Top Rated"
					badgeColor="blue"
					viewAllHref={`/${locale}/${channel}/products?min_rating=4.5`}
					products={topRatedProducts}
					locale={locale}
					channel={channel}
				/>
			)}

			{/* 9. Popular Brands Section */}
			<PopularBrands locale={locale} channel={channel} />

			{/* 10. Value Propositions */}
			<MulticolumnSection
				heading={values.heading}
				columns={valueColumns}
				columnsDesktop={values.columnsDesktop}
				tone="muted"
			/>

			{/* 11. Brand Story */}
			<RichTextBlock
				heading={brandStory.heading}
				paragraphs={brandStory.paragraphs}
				align="center"
				width="narrow"
				tone="inverse"
			/>
		</>
	);
}

export default function Page({ params }: { params: HomeParams }) {
	return (
		<Suspense fallback={<HomePageFallback />}>
			<HomePageContent params={params} />
		</Suspense>
	);
}

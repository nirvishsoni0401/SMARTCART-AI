import type { ListingProvider } from "@/lib/listing/provider";
import type {
  ListingQuery,
  ListingResult,
  ProductCardView,
  SortId,
  FacetView,
} from "@/lib/storefront/contract/listing";
import { smartCartApi, type SmartCartProduct } from "@/lib/smartcart-api";
import { resolveLocaleFromSlug } from "@/config/locale";
import { buildStorefrontPath } from "@/lib/storefront-path";

const SORTS = [
  "featured",
  "newest",
  "price_asc",
  "price_desc",
  "bestselling",
] as const satisfies readonly SortId[];

function toCardView(p: SmartCartProduct, locale: string, channel: string): ProductCardView {
  const currentPrice = Number(p.discounted_price || p.price) || 0;
  const rating = Number(p.rating) || 0;
  const isHighRating = rating >= 4.5;
  const id = String(p.product_id || p.external_id);
  const title = p.title || p.name;
  const catSlug = p.category_slug || p.category.toLowerCase().replace(/ & /g, "-and-").replace(/ /g, "-");

  return {
    id,
    name: title,
    slug: id,
    brand: p.brand && p.brand !== "Unknown" ? p.brand : p.category,
    price: currentPrice,
    currency: "INR",
    localeBcp47: resolveLocaleFromSlug(locale).bcp47,
    image: p.thumbnail || p.image_url || p.images?.[0] || "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=600&q=80",
    hoverImage: p.images?.[1] || null,
    href: buildStorefrontPath(locale, channel, `/products/${id}`),
    badge: p.discount_percentage && p.discount_percentage >= 20 ? `${p.discount_percentage}% OFF` : (isHighRating ? "Top Rated" : null),
    isBestseller: (p.popularity_score || 0) >= 80 || (p.review_count || 0) > 1000,
    category: {
      id: p.category,
      name: p.category,
      slug: catSlug,
    },
    hasVariants: false,
    variantTotalCount: 1,
    variantSampleSize: 1,
  };
}

export const smartcartListingProvider: ListingProvider = {
  id: "smartcart",
  capabilities: {
    surfaces: ["all", "category", "collection", "search"],
    facetCounts: true,
    pagination: "offset",
    sorts: {
      all: SORTS,
      category: SORTS,
      collection: SORTS,
      search: SORTS,
    },
  },
  freshness: { kind: "ttl", profile: "listingTtl" },

  async load(query: ListingQuery): Promise<ListingResult | null> {
    const pageNumber = query.page.mode === "offset" ? query.page.number : 1;
    const pageSize = query.pageSize || 24;
    const skip = (pageNumber - 1) * pageSize;

    // Map sorts
    let sortBy = "default";
    if (query.sort === "price_asc") sortBy = "price_asc";
    else if (query.sort === "price_desc") sortBy = "price_desc";
    else if (query.sort === "bestselling" || query.sort === "featured") sortBy = "popularity";
    else if (query.sort === "newest") sortBy = "newest";

    let categoryParam: string | undefined = undefined;
    if (query.surface.kind === "category") {
      categoryParam = query.surface.slug;
    } else if (query.selections.categories?.length) {
      categoryParam = query.selections.categories[0];
    }

    let brandParam: string | undefined = undefined;
    if (query.selections.brands?.length) {
      brandParam = query.selections.brands[0];
    }

    const minPrice = query.range?.price?.min;
    const maxPrice = query.range?.price?.max;

    let total = 0;
    let products: SmartCartProduct[] = [];

    if (query.surface.kind === "search" && query.surface.text) {
      const searchRes = await smartCartApi.searchProducts(
        query.surface.text,
        pageSize,
        skip,
        pageNumber,
        sortBy
      );
      total = searchRes.total;
      products = searchRes.products;
    } else {
      const prodRes = await smartCartApi.getProducts({
        category: categoryParam,
        brand: brandParam,
        min_price: minPrice,
        max_price: maxPrice,
        sort_by: sortBy,
        limit: pageSize,
        skip,
        page: pageNumber,
      });
      total = prodRes.total;
      products = prodRes.products;
    }

    // Facets (Categories, Subcategories, Brands)
    const [categoriesData, brandsData, subcatsData] = await Promise.all([
      smartCartApi.getCategories().catch(() => ({ categories: [], details: [] })),
      smartCartApi.getBrands(categoryParam).catch(() => ({ brands: [] })),
      smartCartApi.getSubcategories(categoryParam).catch(() => ({ subcategories: [] })),
    ]);

    const activeCatSet = new Set(query.selections.categories || (categoryParam ? [categoryParam] : []));
    const activeBrandSet = new Set(query.selections.brands || []);

    const facets: FacetView[] = [
      {
        id: "categories",
        labelKey: "categories",
        control: "chip",
        values: (categoriesData.details || []).map((cat) => ({
          value: cat.slug,
          label: cat.name,
          count: cat.count,
          selected: activeCatSet.has(cat.slug) || activeCatSet.has(cat.name.toLowerCase()),
        })),
      },
      {
        id: "brands",
        labelKey: "brands",
        control: "chip",
        values: (brandsData.brands || []).slice(0, 20).map((b) => ({
          value: b,
          label: b,
          selected: activeBrandSet.has(b),
        })),
      },
    ];

    const items = products.map((p) => toCardView(p, query.locale, query.channel));

    return {
      items,
      facets,
      sorts: SORTS,
      page: {
        mode: "offset",
        number: pageNumber,
        hasNextPage: skip + pageSize < total,
        hasPreviousPage: pageNumber > 1,
      },
      total: { value: total, exact: true },
    };
  },
};

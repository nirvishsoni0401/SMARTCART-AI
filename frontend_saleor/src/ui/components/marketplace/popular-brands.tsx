import Link from "next/link";
import { Award } from "lucide-react";

const TOP_BRANDS = [
  { name: "Apple", category: "Electronics", count: "Smartphones & Laptops" },
  { name: "Samsung", category: "Electronics", count: "Galaxy & Monitors" },
  { name: "Nike", category: "Fashion", count: "Sportswear & Shoes" },
  { name: "Adidas", category: "Fashion", count: "Athletic & Originals" },
  { name: "Sony", category: "Electronics", count: "Audio & Consoles" },
  { name: "Dell", category: "Electronics", count: "Laptops & Displays" },
  { name: "Prestige", category: "Home & Kitchen", count: "Cookware & Appliances" },
  { name: "Philips", category: "Beauty & Home", count: "Grooming & Kitchen" },
  { name: "boAt", category: "Electronics", count: "Headphones & Audio" },
  { name: "Levi's", category: "Fashion", count: "Denim & Apparel" },
  { name: "Titan", category: "Fashion", count: "Chronograph Watches" },
  { name: "Decathlon", category: "Sports & Fitness", count: "Gym & Outdoor" },
];

interface PopularBrandsProps {
  locale?: string;
  channel?: string;
}

export function PopularBrands({
  locale = "default-channel",
  channel = "in",
}: PopularBrandsProps) {
  return (
    <section className="container-content py-12">
      <div className="mb-6 flex flex-col justify-between sm:flex-row sm:items-end">
        <div>
          <div className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-3 py-0.5 text-xs font-bold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 uppercase">
            <Award className="h-3.5 w-3.5" />
            Official Brands
          </div>
          <h2 className="mt-1 text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl">
            Popular & Trusted Brands
          </h2>
          <p className="text-sm text-muted-foreground">
            Authentic, brand-certified warranty on 120+ leading Indian and international labels.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        {TOP_BRANDS.map((b) => (
          <Link
            key={b.name}
            href={`/${locale}/${channel}/products?brand=${encodeURIComponent(b.name)}`}
            className="group flex flex-col items-center justify-center rounded-xl border bg-card p-4 text-center transition-all duration-200 hover:border-primary/50 hover:shadow-md"
          >
            <span className="text-base font-black tracking-wide text-foreground group-hover:text-primary">
              {b.name}
            </span>
            <span className="mt-1 text-[11px] font-semibold text-primary/80">
              {b.category}
            </span>
            <span className="mt-0.5 text-[10px] text-muted-foreground">
              {b.count}
            </span>
          </Link>
        ))}
      </div>
    </section>
  );
}


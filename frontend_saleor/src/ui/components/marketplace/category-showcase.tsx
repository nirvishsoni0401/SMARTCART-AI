import Link from "next/link";
import {
  Smartphone,
  Shirt,
  Home,
  Sparkles,
  Dumbbell,
  BookOpen,
  Car,
  Gamepad2,
  Apple,
  Briefcase,
  Layers
} from "lucide-react";
import type { CategoryDetail } from "@/lib/smartcart-api";

interface CategoryShowcaseProps {
  categories: CategoryDetail[];
  locale?: string;
  channel?: string;
}

const CATEGORY_ICONS: Record<string, any> = {
  electronics: Smartphone,
  fashion: Shirt,
  "home-and-kitchen": Home,
  "home & kitchen": Home,
  "beauty-and-personal-care": Sparkles,
  "beauty & personal care": Sparkles,
  "sports-and-fitness": Dumbbell,
  "sports & fitness": Dumbbell,
  books: BookOpen,
  automotive: Car,
  "toys-and-baby": Gamepad2,
  "toys & baby products": Gamepad2,
  "grocery-and-food": Apple,
  "grocery & food": Apple,
  "office-and-stationery": Briefcase,
  "office & stationery": Briefcase,
};

export function CategoryShowcase({
  categories,
  locale = "default-channel",
  channel = "in",
}: CategoryShowcaseProps) {
  if (!categories || categories.length === 0) return null;

  return (
    <section className="container-content py-12">
      <div className="mb-6 flex flex-col justify-between sm:flex-row sm:items-end">
        <div>
          <div className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-3 py-0.5 text-xs font-bold text-primary uppercase">
            <Layers className="h-3.5 w-3.5" />
            Mega Marketplace
          </div>
          <h2 className="mt-1 text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl">
            Shop by Category
          </h2>
          <p className="text-sm text-muted-foreground">
            Explore 10,000+ top-rated items curated across 10 diverse retail departments.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-5">
        {categories.map((cat) => {
          const key = cat.slug.toLowerCase();
          const IconComponent = CATEGORY_ICONS[key] || CATEGORY_ICONS[cat.name.toLowerCase()] || Layers;

          return (
            <Link
              key={cat.slug}
              href={`/${locale}/${channel}/categories/${cat.slug}`}
              className="group relative flex flex-col items-center justify-center rounded-2xl border bg-card p-5 text-center transition-all duration-200 hover:-translate-y-1 hover:border-primary/60 hover:shadow-lg"
            >
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary transition-transform duration-300 group-hover:scale-110 group-hover:bg-primary group-hover:text-primary-foreground">
                <IconComponent className="h-7 w-7" />
              </div>

              <h3 className="mt-3 text-sm font-bold text-foreground group-hover:text-primary">
                {cat.name}
              </h3>

              <span className="mt-1 text-xs text-muted-foreground">
                {cat.count.toLocaleString()} Products
              </span>

              {cat.subcategories && cat.subcategories.length > 0 && (
                <div className="mt-2.5 flex flex-wrap justify-center gap-1 opacity-80">
                  {cat.subcategories.slice(0, 2).map((sub) => (
                    <span
                      key={sub}
                      className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
                    >
                      {sub}
                    </span>
                  ))}
                  {cat.subcategories.length > 2 && (
                    <span className="text-[10px] text-muted-foreground">
                      +{cat.subcategories.length - 2} more
                    </span>
                  )}
                </div>
              )}
            </Link>
          );
        })}
      </div>
    </section>
  );
}


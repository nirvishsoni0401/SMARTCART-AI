"use client";

import { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { ShoppingBag, Star, Zap, Check, ArrowRight } from "lucide-react";
import { formatINR, smartCartApi, type SmartCartProduct } from "@/lib/smartcart-api";
import { Button } from "@/ui/components/ui/button";

interface MarketplaceCarouselProps {
  title: string;
  subtitle?: string;
  badgeText?: string;
  badgeColor?: "amber" | "rose" | "blue" | "emerald";
  viewAllHref?: string;
  products: SmartCartProduct[];
  locale?: string;
  channel?: string;
}

export function MarketplaceCarousel({
  title,
  subtitle,
  badgeText,
  badgeColor = "amber",
  viewAllHref = "/products",
  products,
  locale = "default-channel",
  channel = "in",
}: MarketplaceCarouselProps) {
  const [addedState, setAddedState] = useState<Record<number, boolean>>({});

  const handleAddToCart = async (product: SmartCartProduct) => {
    const id = product.product_id || product.external_id;
    setAddedState((prev) => ({ ...prev, [id]: true }));
    try {
      await smartCartApi.addToCart(id, 1);
      setTimeout(() => {
        setAddedState((prev) => ({ ...prev, [id]: false }));
      }, 2000);
    } catch (err) {
      console.error("Failed adding to cart:", err);
      setAddedState((prev) => ({ ...prev, [id]: false }));
    }
  };

  if (!products || products.length === 0) return null;

  const badgeColorClasses = {
    amber: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
    rose: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
    blue: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
    emerald: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  }[badgeColor];

  return (
    <section className="container-content py-12">
      {/* Section Header */}
      <div className="mb-6 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div>
          {badgeText && (
            <div className={`inline-flex items-center gap-1.5 rounded-full px-3 py-0.5 text-xs font-bold tracking-wide uppercase ${badgeColorClasses}`}>
              <Zap className="h-3 w-3" />
              {badgeText}
            </div>
          )}
          <h2 className="mt-1 text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl">
            {title}
          </h2>
          {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
        </div>

        {viewAllHref && (
          <Link
            href={viewAllHref}
            className="group inline-flex items-center text-sm font-semibold text-primary hover:underline"
          >
            Explore All <ArrowRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Link>
        )}
      </div>

      {/* Product Grid / Row */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        {products.slice(0, 12).map((product) => {
          const id = product.product_id || product.external_id;
          const isAdded = addedState[id];
          const hasDiscount = product.discount_percentage && product.discount_percentage > 0;
          const currentPrice = product.discounted_price || product.price;

          return (
            <div
              key={id}
              className="group relative flex flex-col justify-between rounded-xl border bg-card p-3 transition-all duration-200 hover:border-primary/50 hover:shadow-md"
            >
              <div>
                {/* Product Thumbnail */}
                <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-muted/40">
                  <Image
                    src={product.thumbnail || product.image_url || product.images?.[0] || "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=600&q=80"}
                    alt={product.title}
                    fill
                    sizes="(max-width: 640px) 50vw, (max-width: 1024px) 25vw, 16vw"
                    className="object-contain p-2 transition-transform duration-300 group-hover:scale-105"
                    loading="lazy"
                  />

                  {hasDiscount && (
                    <div className="absolute left-1.5 top-1.5 rounded-md bg-rose-600 px-1.5 py-0.5 text-[10px] font-extrabold text-white shadow-sm">
                      {product.discount_percentage}% OFF
                    </div>
                  )}

                  {product.availability === "Low Stock" && (
                    <div className="absolute bottom-1.5 left-1.5 rounded bg-amber-500/90 px-1 py-0.5 text-[9px] font-semibold text-white">
                      Few Left
                    </div>
                  )}
                </div>

                {/* Info */}
                <div className="mt-2.5">
                  <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                    <span className="font-semibold uppercase truncate">{product.brand}</span>
                    <span className="flex items-center gap-0.5 font-bold text-amber-500">
                      <Star className="h-3 w-3 fill-amber-400 text-amber-400" />
                      {product.rating?.toFixed(1) || "4.2"}
                    </span>
                  </div>

                  <h3 className="mt-1 line-clamp-2 text-xs font-semibold text-foreground group-hover:text-primary">
                    <Link href={`/${locale}/${channel}/products/${id}`}>
                      {product.title}
                    </Link>
                  </h3>

                  {/* Price Block */}
                  <div className="mt-2 flex items-baseline gap-1.5">
                    <span className="text-sm font-extrabold text-foreground">
                      {formatINR(currentPrice)}
                    </span>
                    {hasDiscount && (
                      <span className="text-[11px] text-muted-foreground line-through">
                        {formatINR(product.price)}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Quick Add To Cart */}
              <div className="mt-3">
                <Button
                  size="sm"
                  variant={isAdded ? "secondary" : "outline-solid"}
                  onClick={() => handleAddToCart(product)}
                  className="h-8 w-full text-xs font-semibold"
                >
                  {isAdded ? (
                    <>
                      <Check className="mr-1 h-3 w-3 text-emerald-600" /> Added
                    </>
                  ) : (
                    <>
                      <ShoppingBag className="mr-1 h-3 w-3" /> Add
                    </>
                  )}
                </Button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { Plus, ShoppingBag, Check, Star } from "lucide-react";
import { smartCartApi, formatINR, type SmartCartProduct } from "@/lib/smartcart-api";
import { Button } from "@/ui/components/ui/button";

interface PdpCrossSellProps {
  productId: string | number;
  locale?: string;
  channel?: string;
}

export function PdpCrossSell({
  productId,
  locale = "default-channel",
  channel = "in",
}: PdpCrossSellProps) {
  const [fbt, setFbt] = useState<SmartCartProduct[]>([]);
  const [similar, setSimilar] = useState<SmartCartProduct[]>([]);
  const [bundleAdded, setBundleAdded] = useState(false);
  const [addedIds, setAddedIds] = useState<Record<number, boolean>>({});

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        const [fbtRes, simRes] = await Promise.all([
          smartCartApi.getFrequentlyBoughtTogether(productId, 3).catch(() => ({ products: [] })),
          smartCartApi.getSimilarProducts(productId, 6).catch(() => ({ products: [] })),
        ]);
        if (isMounted) {
          setFbt(fbtRes.products || []);
          setSimilar(simRes.products || []);
        }
      } catch (err) {
        console.error("Error loading cross-sell products:", err);
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, [productId]);

  const handleAddBundle = async () => {
    setBundleAdded(true);
    try {
      // Add main product + all FBT items
      const pNum = Number(productId);
      if (!isNaN(pNum)) {
        await smartCartApi.addToCart(pNum, 1);
      }
      for (const item of fbt) {
        const itemId = item.product_id || item.external_id;
        await smartCartApi.addToCart(itemId, 1);
      }
      setTimeout(() => setBundleAdded(false), 2500);
    } catch (err) {
      console.error("Failed adding bundle:", err);
      setBundleAdded(false);
    }
  };

  const handleAddSingle = async (p: SmartCartProduct) => {
    const id = p.product_id || p.external_id;
    setAddedIds((prev) => ({ ...prev, [id]: true }));
    try {
      await smartCartApi.addToCart(id, 1);
      setTimeout(() => setAddedIds((prev) => ({ ...prev, [id]: false })), 2000);
    } catch (err) {
      console.error("Failed adding item:", err);
      setAddedIds((prev) => ({ ...prev, [id]: false }));
    }
  };

  if (fbt.length === 0 && similar.length === 0) return null;

  const bundleTotal = fbt.reduce(
    (sum, item) => sum + (item.discounted_price || item.price),
    0
  );

  return (
    <div className="container-content my-12 space-y-12">
      {/* 1. Frequently Bought Together */}
      {fbt.length > 0 && (
        <section className="rounded-2xl border bg-card p-6 shadow-sm">
          <div className="mb-4">
            <span className="text-xs font-bold uppercase tracking-wider text-primary">
              Recommended Combo
            </span>
            <h2 className="text-xl font-extrabold text-foreground">
              Frequently Bought Together
            </h2>
          </div>

          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            {/* Products Row */}
            <div className="flex flex-wrap items-center gap-4">
              {fbt.map((item, idx) => {
                const id = item.product_id || item.external_id;
                return (
                  <div key={id} className="flex items-center gap-4">
                    <div className="flex w-36 flex-col rounded-xl border bg-muted/20 p-2.5">
                      <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-background">
                        <Image
                          src={item.thumbnail || item.image_url || item.images?.[0] || "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=300&q=80"}
                          alt={item.title}
                          fill
                          className="object-contain p-2"
                        />
                      </div>
                      <Link
                        href={`/${locale}/${channel}/products/${id}`}
                        className="mt-2 line-clamp-1 text-xs font-semibold text-foreground hover:underline"
                      >
                        {item.title}
                      </Link>
                      <span className="text-xs font-bold text-foreground">
                        {formatINR(item.discounted_price || item.price)}
                      </span>
                    </div>

                    {idx < fbt.length - 1 && (
                      <div className="rounded-full bg-muted p-1 text-muted-foreground">
                        <Plus className="h-4 w-4" />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Bundle Action */}
            <div className="flex flex-col items-start gap-3 rounded-xl border bg-muted/40 p-5 lg:w-64">
              <div>
                <span className="text-xs text-muted-foreground">Bundle Total:</span>
                <div className="text-xl font-extrabold text-foreground">
                  {formatINR(bundleTotal)}
                </div>
              </div>
              <Button
                onClick={handleAddBundle}
                className="w-full font-semibold"
                variant={bundleAdded ? "secondary" : "default"}
              >
                {bundleAdded ? (
                  <>
                    <Check className="mr-1.5 h-4 w-4 text-emerald-600" /> Bundle Added
                  </>
                ) : (
                  <>
                    <ShoppingBag className="mr-1.5 h-4 w-4" /> Add Bundle to Cart
                  </>
                )}
              </Button>
            </div>
          </div>
        </section>
      )}

      {/* 2. Similar Products */}
      {similar.length > 0 && (
        <section>
          <div className="mb-6 flex items-center justify-between">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
                Compare & Discover
              </span>
              <h2 className="text-xl font-extrabold text-foreground sm:text-2xl">
                Similar Products in this Category
              </h2>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
            {similar.map((p) => {
              const id = p.product_id || p.external_id;
              const isAdded = addedIds[id];
              return (
                <div
                  key={id}
                  className="group relative flex flex-col justify-between rounded-xl border bg-card p-3 transition hover:border-primary/50 hover:shadow-md"
                >
                  <div>
                    <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-muted/30">
                      <Image
                        src={p.thumbnail || p.image_url || p.images?.[0] || "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=300&q=80"}
                        alt={p.title}
                        fill
                        className="object-contain p-2 group-hover:scale-105 transition-transform"
                      />
                    </div>
                    <div className="mt-2">
                      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                        <span className="font-semibold uppercase">{p.brand}</span>
                        <span className="flex items-center gap-0.5 font-bold text-amber-500">
                          <Star className="h-3 w-3 fill-amber-400 text-amber-400" />
                          {p.rating?.toFixed(1) || "4.3"}
                        </span>
                      </div>
                      <h3 className="mt-1 line-clamp-2 text-xs font-semibold text-foreground group-hover:text-primary">
                        <Link href={`/${locale}/${channel}/products/${id}`}>
                          {p.title}
                        </Link>
                      </h3>
                      <div className="mt-1 font-bold text-sm text-foreground">
                        {formatINR(p.discounted_price || p.price)}
                      </div>
                    </div>
                  </div>

                  <div className="mt-2.5">
                    <Button
                      size="sm"
                      variant={isAdded ? "secondary" : "outline-solid"}
                      onClick={() => handleAddSingle(p)}
                      className="h-7 w-full text-xs font-semibold"
                    >
                      {isAdded ? "Added" : "Add to Cart"}
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

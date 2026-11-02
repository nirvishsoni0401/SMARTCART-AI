"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Flame, Clock, ArrowRight } from "lucide-react";

interface DealsBannerProps {
  locale?: string;
  channel?: string;
}

export function DealsBanner({
  locale = "default-channel",
  channel = "in",
}: DealsBannerProps) {
  const [timeLeft, setTimeLeft] = useState({ hours: 7, minutes: 42, seconds: 19 });

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev.seconds > 0) return { ...prev, seconds: prev.seconds - 1 };
        if (prev.minutes > 0) return { ...prev, minutes: 59, seconds: 59 };
        if (prev.hours > 0) return { hours: prev.hours - 1, minutes: 59, seconds: 59 };
        return { hours: 12, minutes: 0, seconds: 0 };
      });
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="container-content my-6">
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-red-600 via-rose-600 to-amber-600 p-6 text-white shadow-xl sm:p-8">
        <div className="relative z-10 flex flex-col items-start justify-between gap-6 md:flex-row md:items-center">
          <div>
            <div className="inline-flex items-center gap-1.5 rounded-full bg-white/20 px-3 py-1 text-xs font-black tracking-wider uppercase backdrop-blur-sm">
              <Flame className="h-4 w-4 fill-amber-300 text-amber-300" />
              Limited-Time Super Deals
            </div>
            <h2 className="mt-3 text-3xl font-black tracking-tight sm:text-4xl">
              Save Up to 65% OFF Today
            </h2>
            <p className="mt-1 text-sm text-white/90 max-w-xl">
              Special discounts on Smartphones, Laptops, Fashion Apparel, Cookware & Home Essentials.
            </p>
          </div>

          <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center">
            {/* Countdown timer */}
            <div className="flex items-center gap-2 rounded-xl bg-black/30 p-2.5 backdrop-blur-md">
              <Clock className="h-4 w-4 text-amber-300" />
              <div className="flex items-center gap-1 font-mono text-sm font-bold">
                <span className="rounded bg-black/50 px-2 py-1">
                  {String(timeLeft.hours).padStart(2, "0")}h
                </span>
                <span>:</span>
                <span className="rounded bg-black/50 px-2 py-1">
                  {String(timeLeft.minutes).padStart(2, "0")}m
                </span>
                <span>:</span>
                <span className="rounded bg-black/50 px-2 py-1">
                  {String(timeLeft.seconds).padStart(2, "0")}s
                </span>
              </div>
            </div>

            <Link
              href={`/${locale}/${channel}/products?sort_by=discount_desc`}
              className="inline-flex items-center rounded-xl bg-white px-5 py-3 text-sm font-bold text-red-700 shadow-md transition-all hover:bg-amber-50 hover:shadow-lg"
            >
              Shop Deals <ArrowRight className="ml-1.5 h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}


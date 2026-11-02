"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { Sparkles, ThumbsUp, ThumbsDown, ShoppingBag, EyeOff, Check } from "lucide-react";
import { smartCartApi, formatINR, type RecommendationItem } from "@/lib/smartcart-api";
import { Button } from "@/ui/components/ui/button";

interface RecommendationSectionProps {
	locale: string;
	channel: string;
}

export function RecommendationSection({ locale, channel }: RecommendationSectionProps) {
	const [items, setItems] = useState<RecommendationItem[]>([]);
	const [recommendationId, setRecommendationId] = useState<string | null>(null);
	const [feedbackState, setFeedbackState] = useState<Record<number, "liked" | "disliked" | "dismissed">>({});
	const [addedToCartState, setAddedToCartState] = useState<Record<number, boolean>>({});
	const [isLoading, setIsLoading] = useState(true);

	useEffect(() => {
		let isMounted = true;
		async function fetchRecs() {
			try {
				const data = await smartCartApi.getRecommendations(8);
				if (isMounted) {
					setItems(data.recommendations || []);
					setRecommendationId(data.recommendation_id);
					// Track impressions for the bandit
					for (const item of (data.recommendations || []).slice(0, 4)) {
						smartCartApi.trackEvent({
							event_type: "product_impression",
							product_id: item.external_id,
							recommendation_id: data.recommendation_id,
						});
					}
				}
			} catch (e) {
				console.error("Failed to load recommendations:", e);
			} finally {
				if (isMounted) setIsLoading(false);
			}
		}

		fetchRecs();
		return () => {
			isMounted = false;
		};
	}, []);

	const handleFeedback = async (productId: number, action: "like" | "dislike" | "not_interested") => {
		setFeedbackState((prev) => ({
			...prev,
			[productId]: action === "like" ? "liked" : action === "dislike" ? "disliked" : "dismissed",
		}));

		try {
			await smartCartApi.sendRecommendationFeedback({
				recommendation_id: recommendationId || undefined,
				product_id: productId,
				action,
			});
		} catch (e) {
			console.error("Failed to submit feedback:", e);
		}
	};

	const handleAddToCart = async (product: RecommendationItem) => {
		setAddedToCartState((prev) => ({ ...prev, [product.external_id]: true }));
		try {
			await smartCartApi.addToCart(product.external_id, 1);
			await smartCartApi.sendRecommendationFeedback({
				recommendation_id: recommendationId || undefined,
				product_id: product.external_id,
				action: "cart",
			});
			setTimeout(() => {
				setAddedToCartState((prev) => ({ ...prev, [product.external_id]: false }));
			}, 2500);
		} catch (e) {
			console.error("Failed to add recommended product to cart:", e);
		}
	};

	const handleProductClick = (productId: number) => {
		smartCartApi.trackEvent({
			event_type: "product_click",
			product_id: productId,
			recommendation_id: recommendationId || undefined,
		});
	};

	if (isLoading) {
		return (
			<section className="container-content py-12" id="recommended">
				<div className="mb-8 flex items-center gap-2">
					<div className="h-6 w-6 animate-pulse rounded bg-muted" />
					<div className="h-8 w-64 animate-pulse rounded bg-muted" />
				</div>
				<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
					{[1, 2, 3, 4].map((i) => (
						<div key={i} className="h-80 animate-pulse rounded-2xl border bg-muted/40 p-4" />
					))}
				</div>
			</section>
		);
	}

	const visibleItems = items.filter((item) => feedbackState[item.external_id] !== "dismissed");

	if (visibleItems.length === 0) {
		return null;
	}

	return (
		<section className="container-content py-16" id="recommended">
			{/* Section Header with AI Badge */}
			<div className="mb-10 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
				<div>
					<div className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700 dark:bg-blue-950 dark:text-blue-300">
						<Sparkles className="h-3.5 w-3.5" />
						Adaptive LinUCB Bandits
					</div>
					<h2 className="mt-2 text-2xl font-extrabold tracking-tight text-foreground sm:text-3xl">
						Recommended for You
					</h2>
					<p className="mt-1 text-sm text-muted-foreground">
						Personalized real-time suggestions that continuously adapt to your browsing preferences and feedback.
					</p>
				</div>
				<div className="text-xs text-muted-foreground">
					Click <ThumbsUp className="inline h-3.5 w-3.5 text-emerald-600" /> or{" "}
					<ThumbsDown className="inline h-3.5 w-3.5 text-red-600" /> to tune your AI agent
				</div>
			</div>

			{/* Products Grid */}
			<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
				{visibleItems.map((item) => {
					const isLiked = feedbackState[item.external_id] === "liked";
					const isDisliked = feedbackState[item.external_id] === "disliked";
					const isAdded = addedToCartState[item.external_id];

					return (
						<div
							key={item.external_id}
							className="group relative flex flex-col justify-between rounded-2xl border bg-card p-4 transition-all duration-300 hover:border-primary/40 hover:shadow-lg"
						>
							<div>
								{/* Thumbnail */}
								<div className="relative aspect-square w-full overflow-hidden rounded-xl border bg-muted/30">
									{item.image_url ? (
										<Image
											src={item.image_url}
											alt={item.name}
											fill
											className="object-contain p-4 transition-transform duration-300 group-hover:scale-105"
										/>
									) : (
										<div className="flex h-full w-full items-center justify-center text-xs text-muted-foreground">
											No image
										</div>
									)}

									{/* Adaptive Explanation Pill */}
									<div className="absolute bottom-2 left-2 right-2 rounded-md bg-background/90 px-2 py-1 text-center text-[11px] font-medium text-foreground backdrop-blur-sm shadow-sm">
										{item.explanation}
									</div>
								</div>

								{/* Category & Title */}
								<div className="mt-4">
									<div className="flex items-center justify-between text-xs text-muted-foreground">
										<span className="font-semibold uppercase tracking-wider">{item.category}</span>
										<span className="flex items-center font-bold text-amber-500">
											★ {item.rating?.toFixed(1) || "4.5"}
										</span>
									</div>
									<h3 className="mt-1 text-sm font-bold text-foreground line-clamp-2">
										<Link
											href={`/${locale}/${channel}/products/${item.product_id || item.external_id}`}
											onClick={() => handleProductClick(item.product_id || item.external_id)}
											className="hover:underline"
										>
											{item.title || item.name}
										</Link>
									</h3>
									<div className="mt-2 flex items-baseline gap-2">
										<span className="text-lg font-extrabold text-foreground">
											{formatINR(item.discounted_price || item.price)}
										</span>
										{item.discount_percentage && item.discount_percentage > 0 && (
											<span className="text-xs text-muted-foreground line-through">
												{formatINR(item.price)}
											</span>
										)}
									</div>
								</div>
							</div>

							{/* Feedback Controls & Add to Cart */}
							<div className="mt-4 border-t pt-3">
								<div className="mb-2.5 flex items-center justify-between text-xs">
									<span className="text-[11px] text-muted-foreground">Help AI learn:</span>
									<div className="flex items-center gap-1">
										<button
											onClick={() => handleFeedback(item.external_id, "like")}
											title="Recommend more like this"
											className={`rounded p-1 transition ${
												isLiked ? "bg-emerald-100 text-emerald-700" : "text-muted-foreground hover:bg-muted hover:text-foreground"
											}`}
										>
											<ThumbsUp className="h-3.5 w-3.5" />
										</button>
										<button
											onClick={() => handleFeedback(item.external_id, "dislike")}
											title="Not interested / Dislike"
											className={`rounded p-1 transition ${
												isDisliked ? "bg-red-100 text-red-700" : "text-muted-foreground hover:bg-muted hover:text-foreground"
											}`}
										>
											<ThumbsDown className="h-3.5 w-3.5" />
										</button>
										<button
											onClick={() => handleFeedback(item.external_id, "not_interested")}
											title="Hide product"
											className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition"
										>
											<EyeOff className="h-3.5 w-3.5" />
										</button>
									</div>
								</div>

								<Button
									size="sm"
									onClick={() => handleAddToCart(item)}
									className="w-full text-xs font-semibold"
									variant={isAdded ? "secondary" : "default"}
								>
									{isAdded ? (
										<>
											<Check className="mr-1.5 h-3.5 w-3.5 text-emerald-600" /> Added to Cart
										</>
									) : (
										<>
											<ShoppingBag className="mr-1.5 h-3.5 w-3.5" /> Add to Cart
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


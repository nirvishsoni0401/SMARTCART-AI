"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { SearchIcon, X, Tag, ArrowRight, Sparkles } from "lucide-react";
import { smartCartApi, formatINR, type SearchSuggestionsResponse } from "@/lib/smartcart-api";
import { buildStorefrontPath } from "@/lib/storefront-path";

export const SearchBar = ({
	locale,
	channel,
	placeholder,
	srOnlyLabel,
}: {
	locale: string;
	channel: string;
	placeholder: string;
	srOnlyLabel: string;
}) => {
	const router = useRouter();
	const [query, setQuery] = useState("");
	const [suggestions, setSuggestions] = useState<SearchSuggestionsResponse | null>(null);
	const [isOpen, setIsOpen] = useState(false);
	const [isLoading, setIsLoading] = useState(false);
	const containerRef = useRef<HTMLDivElement>(null);

	// Debounced suggestion fetcher
	useEffect(() => {
		if (!query || query.trim().length < 2) {
			setSuggestions(null);
			setIsLoading(false);
			return;
		}

		setIsLoading(true);
		const timer = setTimeout(async () => {
			try {
				const data = await smartCartApi.getSearchSuggestions(query.trim());
				setSuggestions(data);
				setIsOpen(true);
			} catch (err) {
				console.error("Suggestions fetch error:", err);
			} finally {
				setIsLoading(false);
			}
		}, 180);

		return () => clearTimeout(timer);
	}, [query]);

	// Close dropdown when clicking outside
	useEffect(() => {
		function handleClickOutside(e: MouseEvent) {
			if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
				setIsOpen(false);
			}
		}
		document.addEventListener("mousedown", handleClickOutside);
		return () => document.removeEventListener("mousedown", handleClickOutside);
	}, []);

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		if (!query.trim()) return;
		setIsOpen(false);
		router.push(`${buildStorefrontPath(locale, channel, "/search")}?query=${encodeURIComponent(query.trim())}`);
	};

	const hasResults =
		suggestions &&
		(suggestions.products.length > 0 ||
			suggestions.brands.length > 0 ||
			suggestions.subcategories.length > 0);

	return (
		<div ref={containerRef} className="group relative w-full max-w-md">
			<form onSubmit={handleSubmit} className="relative block">
				<span className="sr-only">{srOnlyLabel}</span>
				{/* Search icon */}
				<span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-4">
					<SearchIcon
						className="h-4 w-4 text-muted-foreground transition-colors group-focus-within:text-foreground"
						aria-hidden
					/>
				</span>

				{/* Input */}
				<input
					type="text"
					name="search"
					value={query}
					onChange={(e) => {
						setQuery(e.target.value);
						if (!isOpen && e.target.value.trim().length >= 2) setIsOpen(true);
					}}
					onFocus={() => {
						if (query.trim().length >= 2) setIsOpen(true);
					}}
					placeholder={placeholder}
					autoComplete="off"
					className="hover:bg-secondary/80 focus:outline-hidden h-10 w-full rounded-lg border border-transparent bg-secondary py-2 pl-11 pr-10 text-sm text-foreground transition-all placeholder:text-muted-foreground hover:border-border focus:border-ring focus:bg-background focus:ring-1 focus:ring-ring"
				/>

				{/* Clear / Spinner Button */}
				{query && (
					<button
						type="button"
						onClick={() => {
							setQuery("");
							setSuggestions(null);
							setIsOpen(false);
						}}
						className="absolute inset-y-0 right-0 flex items-center pr-3 text-muted-foreground hover:text-foreground"
					>
						<X className="h-4 w-4" />
					</button>
				)}
			</form>

			{/* Autocomplete Dropdown */}
			{isOpen && hasResults && (
				<div className="absolute left-0 right-0 top-full z-50 mt-1 max-h-[480px] overflow-y-auto rounded-xl border bg-popover p-3 text-popover-foreground shadow-2xl animate-in fade-in-50 zoom-in-95">
					{/* Brands & Subcategories Tags */}
					{(suggestions.brands.length > 0 || suggestions.subcategories.length > 0) && (
						<div className="mb-3 border-b pb-2.5">
							<span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
								Quick Filters
							</span>
							<div className="mt-1.5 flex flex-wrap gap-1.5">
								{suggestions.brands.map((brand) => (
									<Link
										key={brand}
										href={`${buildStorefrontPath(locale, channel, "/products")}?brand=${encodeURIComponent(brand)}`}
										onClick={() => setIsOpen(false)}
										className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs font-semibold text-foreground hover:bg-primary hover:text-primary-foreground transition-colors"
									>
										<Tag className="h-3 w-3" />
										{brand}
									</Link>
								))}
								{suggestions.subcategories.map((sub) => (
									<Link
										key={sub}
										href={`${buildStorefrontPath(locale, channel, "/products")}?category=${encodeURIComponent(sub)}`}
										onClick={() => setIsOpen(false)}
										className="inline-flex items-center gap-1 rounded-md bg-secondary px-2 py-1 text-xs font-semibold text-secondary-foreground hover:bg-primary hover:text-primary-foreground transition-colors"
									>
										<Sparkles className="h-3 w-3" />
										{sub}
									</Link>
								))}
							</div>
						</div>
					)}

					{/* Matching Products */}
					{suggestions.products.length > 0 && (
						<div>
							<span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
								Matching Products
							</span>
							<div className="mt-1 divide-y divide-border">
								{suggestions.products.map((p) => (
									<Link
										key={p.product_id}
										href={`${buildStorefrontPath(locale, channel, `/products/${p.product_id}`)}`}
										onClick={() => setIsOpen(false)}
										className="flex items-center gap-3 py-2 hover:bg-muted/60 rounded-lg px-2 transition-colors"
									>
										<div className="relative h-11 w-11 shrink-0 overflow-hidden rounded-md border bg-muted/40">
											<Image
												src={p.thumbnail || "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=120&q=80"}
												alt={p.title}
												fill
												className="object-contain p-1"
											/>
										</div>
										<div className="flex-1 min-w-0">
											<p className="text-xs font-semibold text-foreground truncate">
												{p.title}
											</p>
											<div className="flex items-center gap-2 text-[11px] text-muted-foreground">
												<span>{p.category}</span>
												<span>•</span>
												<span className="font-bold text-foreground">
													{formatINR(p.price)}
												</span>
											</div>
										</div>
									</Link>
								))}
							</div>
						</div>
					)}

					{/* View All Footer */}
					<div className="mt-2 border-t pt-2">
						<button
							onClick={handleSubmit}
							className="w-full flex items-center justify-between rounded-lg px-2 py-1.5 text-xs font-bold text-primary hover:bg-primary/10 transition-colors"
						>
							<span>See all results for &ldquo;{query}&rdquo;</span>
							<ArrowRight className="h-3.5 w-3.5" />
						</button>
					</div>
				</div>
			)}
		</div>
	);
};

import { Suspense } from "react";
import { type Metadata } from "next";
import { API_BASE_URL } from "@/lib/smartcart-api";
import { AnalyticsDashboardClient } from "./analytics-dashboard-client";

export const metadata: Metadata = {
	title: "Research Dashboard & Bandit Analytics | SmartCart AI",
	description: "Reinforcement learning recommendation metrics, LinUCB policy updates, and simulation experiments.",
};

export default function AnalyticsPage(props: { params: Promise<{ locale: string; channel: string }> }) {
	return (
		<section className="container-content py-10">
			<Suspense fallback={<DashboardSkeleton />}>
				<DashboardLoader params={props.params} />
			</Suspense>
		</section>
	);
}

async function DashboardLoader({ params: paramsPromise }: { params: Promise<{ locale: string; channel: string }> }) {
	const params = await paramsPromise;

	let overview = null;
	let comparison = null;

	try {
		const [overviewRes, compRes] = await Promise.all([
			fetch(`${API_BASE_URL}/analytics/overview`, { cache: "no-store" }),
			fetch(`${API_BASE_URL}/analytics/comparison`, { cache: "no-store" }),
		]);
		if (overviewRes.ok) overview = await overviewRes.json();
		if (compRes.ok) comparison = await compRes.json();
	} catch (e) {
		console.error("DashboardLoader fetch error:", e);
	}

	return (
		<AnalyticsDashboardClient
			initialOverview={overview}
			initialComparison={comparison}
			locale={params.locale}
			channel={params.channel}
		/>
	);
}

function DashboardSkeleton() {
	return (
		<div className="animate-pulse space-y-6">
			<div className="h-10 w-72 rounded bg-muted" />
			<div className="grid grid-cols-2 gap-4 md:grid-cols-4">
				{[1, 2, 3, 4].map((i) => (
					<div key={i} className="h-28 rounded-xl bg-muted" />
				))}
			</div>
			<div className="h-96 rounded-xl bg-muted" />
		</div>
	);
}


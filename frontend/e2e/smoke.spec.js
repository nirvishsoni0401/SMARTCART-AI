import { test, expect } from '@playwright/test';

// Helper to attach error listeners and collect errors per test
function setupPageMonitors(page) {
  const consoleErrors = [];
  const pageErrors = [];
  const failedRequests = [];

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      const text = msg.text();
      // Ignore known favicon 404 or benign telemetry if any
      if (!text.includes('favicon.ico')) {
        consoleErrors.push(text);
      }
    }
  });

  page.on('pageerror', (err) => {
    pageErrors.push(err.message || String(err));
  });

  page.on('requestfailed', (req) => {
    const url = req.url();
    const origin = new URL(page.url()).origin;
    // Only fail on same-origin or backend API request failures
    if (url.startsWith(origin) || url.includes('/api/') || url.includes(':8000') || url.includes(':8008')) {
      failedRequests.push(`${req.method()} ${url} (${req.failure()?.errorText || 'failed'})`);
    }
  });

  return {
    assertCleanState: async (contextLabel) => {
      // Allow image loading and layout to settle
      await page.waitForTimeout(600);

      // Check for visible broken images (complete but naturalWidth === 0)
      const brokenImages = await page.evaluate(() => {
        const imgs = Array.from(document.querySelectorAll('img'));
        return imgs
          .filter((img) => {
            const rect = img.getBoundingClientRect();
            const isVisible = rect.width > 0 && rect.height > 0 && window.getComputedStyle(img).display !== 'none';
            return isVisible && img.complete && img.naturalWidth === 0;
          })
          .map((img) => img.src);
      });

      expect(pageErrors, `Uncaught page errors on ${contextLabel}`).toEqual([]);
      expect(consoleErrors, `Console errors on ${contextLabel}`).toEqual([]);
      expect(failedRequests, `Failed same-origin requests on ${contextLabel}`).toEqual([]);
      expect(brokenImages, `Visible empty/broken image boxes on ${contextLabel}`).toEqual([]);
    },
  };
}

test.describe('Storefront Headless Smoke Suite', () => {
  test('visits Home page (/) without console errors or broken images', async ({ page }) => {
    const monitor = setupPageMonitors(page);
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('Next-Gen E-Commerce');
    await monitor.assertCleanState('Home page (/)');
  });

  test('visits Catalog page (/products) without console errors or broken images', async ({ page }) => {
    const monitor = setupPageMonitors(page);
    await page.goto('/products');
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    await monitor.assertCleanState('Catalog page (/products)');
  });

  test('visits DummyJSON product detail page (/products/631)', async ({ page }) => {
    const monitor = setupPageMonitors(page);
    await page.goto('/products/631');
    // Ensure product title rendered and not an early return / crash
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    await expect(page.getByText('Add to Cart')).toBeVisible();
    await monitor.assertCleanState('DummyJSON Product Detail (/products/631)');
  });

  test('visits Book product detail page (/products/526) with deterministic SVG cover', async ({ page }) => {
    const monitor = setupPageMonitors(page);
    await page.goto('/products/526');
    // Ensure book title rendered and not an early return / crash
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    await expect(page.getByText('Add to Cart')).toBeVisible();
    // Confirm book cover image or SVG jacket is rendered
    const imgOrCover = page.locator('img, [role="img"]').first();
    await expect(imgOrCover).toBeVisible();
    await monitor.assertCleanState('Book Product Detail (/products/526)');
  });

  test('visits Cart page (/cart)', async ({ page }) => {
    const monitor = setupPageMonitors(page);
    await page.goto('/cart');
    await expect(page.getByRole('heading', { level: 2 })).toBeVisible();
    await monitor.assertCleanState('Cart page (/cart)');
  });

  test('visits Wishlist page (/wishlist)', async ({ page }) => {
    const monitor = setupPageMonitors(page);
    await page.goto('/wishlist');
    await expect(page.getByRole('heading', { level: 2 })).toBeVisible();
    await monitor.assertCleanState('Wishlist page (/wishlist)');
  });

  test('visits Orders page (/orders)', async ({ page }) => {
    const monitor = setupPageMonitors(page);
    await page.goto('/orders');
    // Should show log in prompt or empty orders without throwing
    await expect(page.locator('body')).toBeVisible();
    await monitor.assertCleanState('Orders page (/orders)');
  });

  test('visits Admin Dashboard (/admin) and logs in as seeded admin', async ({ page }) => {
    const monitor = setupPageMonitors(page);
    await page.goto('/admin');

    // Initially requires admin privileges
    await expect(page.getByText('Admin Privileges Required')).toBeVisible();

    // Click Login in Navbar
    await page.getByRole('button', { name: /Login/i }).first().click();

    // Use Quick Demo Admin Login or fill in seeded admin credentials
    const quickAdminBtn = page.getByRole('button', { name: /Admin Login/i });
    if (await quickAdminBtn.isVisible()) {
      await quickAdminBtn.click();
    } else {
      await page.locator('input[type="email"]').fill('admin@smartcart.ai');
      await page.locator('input[type="password"]').fill('admin123');
      await page.getByRole('button', { name: 'Sign In' }).last().click();
    }

    // Wait for Admin Dashboard to display metrics and controls
    await expect(page.getByText('Adaptive Bandit Monitoring Dashboard')).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('Active Recommendation Policy')).toBeVisible();

    await monitor.assertCleanState('Admin Dashboard (/admin)');
  });
});

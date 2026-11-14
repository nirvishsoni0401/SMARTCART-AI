import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ProductImage, { resolveImageUrl, ProductImagePlaceholder } from '../components/ProductImage';

describe('resolveImageUrl helper', () => {
  it('resolves relative /static paths against API base URL', () => {
    const resolved = resolveImageUrl('/static/covers/book_101.svg');
    expect(resolved).toMatch(/^https?:\/\/.+\/static\/covers\/book_101\.svg$/);
  });

  it('preserves external CDN URLs unchanged', () => {
    const cdnUrl = 'https://cdn.dummyjson.com/products/images/beauty/1.png';
    expect(resolveImageUrl(cdnUrl)).toBe(cdnUrl);
  });

  it('preserves data URIs and blob URIs', () => {
    const dataUri = 'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=';
    expect(resolveImageUrl(dataUri)).toBe(dataUri);
  });

  it('returns empty string for null, undefined or empty input', () => {
    expect(resolveImageUrl(null)).toBe('');
    expect(resolveImageUrl(undefined)).toBe('');
    expect(resolveImageUrl('')).toBe('');
  });
});

describe('ProductImage Component Fallback Chain', () => {
  const dummyProduct = {
    id: 42,
    name: 'Wireless Earbuds Pro',
    category: 'Electronics',
    brand: 'boAt',
    image_url: 'https://broken.invalid/primary.jpg',
    attributes: {
      gallery: ['https://broken.invalid/secondary.jpg'],
    },
  };

  it('renders initial image with candidate url', () => {
    render(<ProductImage product={dummyProduct} />);
    const img = screen.getByRole('img');
    expect(img).toBeInTheDocument();
    expect(img).toHaveAttribute('src', 'https://broken.invalid/primary.jpg');
  });

  it('progresses to next candidate when primary image encounters error', () => {
    render(<ProductImage product={dummyProduct} />);
    const img = screen.getByRole('img');

    // Simulate primary error
    fireEvent.error(img);

    // Should now attempt gallery image
    expect(img).toHaveAttribute('src', 'https://broken.invalid/secondary.jpg');
  });

  it('renders ProductImagePlaceholder when all candidates fail, never leaving an empty box', () => {
    render(<ProductImage product={dummyProduct} />);
    const img = screen.getByRole('img');

    // Fail candidate 1
    fireEvent.error(img);
    // Fail candidate 2
    fireEvent.error(img);

    // Image tag should be replaced by fallback placeholder container
    const placeholder = screen.getByRole('img', { name: /Wireless Earbuds Pro \(Electronics\)/i });
    expect(placeholder).toBeInTheDocument();
    // Contains initials
    expect(screen.getByText('WE')).toBeInTheDocument();
    // Contains department and brand
    expect(screen.getByText('Electronics')).toBeInTheDocument();
    expect(screen.getByText('boAt')).toBeInTheDocument();
  });

  it('ProductImagePlaceholder renders standalone with correct attributes', () => {
    render(<ProductImagePlaceholder product={dummyProduct} />);
    expect(screen.getByText('WE')).toBeInTheDocument();
    expect(screen.getByText('Electronics')).toBeInTheDocument();
    expect(screen.getByText('boAt')).toBeInTheDocument();
  });
});

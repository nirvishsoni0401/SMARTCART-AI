"""
SmartCart AI — Deterministic Local Book Cover Generator
Generates clean, elegant vector (SVG) book covers using SHA-256 title hashing.
Zero network dependencies; 100% reproducible and offline-safe.
"""

import html
import hashlib
from pathlib import Path
from typing import Tuple, List

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
COVERS_DIR = STATIC_DIR / "covers"

# 16 Curated High-Contrast Duotone & Gradient Palettes for Book Covers
PALETTES: List[Tuple[str, str, str]] = [
    ("#0f172a", "#1e1b4b", "#38bdf8"),  # Midnight Slate -> Deep Indigo (Cyan accent)
    ("#064e3b", "#022c22", "#34d399"),  # Emerald -> Forest Deep (Mint accent)
    ("#4c0519", "#881337", "#fb7185"),  # Bordeaux -> Deep Rose (Rose accent)
    ("#3b0764", "#581c87", "#c084fc"),  # Imperial Purple -> Royal Violet (Lavender accent)
    ("#134e4a", "#0f172a", "#2dd4bf"),  # Deep Teal -> Dark Slate (Teal accent)
    ("#451a03", "#78350f", "#fbbf24"),  # Warm Mahogany -> Deep Amber (Gold accent)
    ("#172554", "#1e3a8a", "#60a5fa"),  # Deep Navy -> Cobalt (Sky accent)
    ("#701a75", "#4a044e", "#f472b6"),  # Royal Plum -> Dark Berry (Pink accent)
    ("#18181b", "#27272a", "#a1a1aa"),  # Titanium Charcoal -> Zinc (Slate accent)
    ("#431407", "#7c2d12", "#f97316"),  # Terracotta -> Burnt Sienna (Orange accent)
    ("#083344", "#155e75", "#38bdf8"),  # Deep Ocean -> Cyan Dusk (Light Blue accent)
    ("#2e1065", "#1e1b4b", "#a855f7"),  # Night Violet -> Ultra Violet (Purple accent)
    ("#1c1917", "#292524", "#e7e5e4"),  # Warm Espresso -> Deep Stone (Warm White accent)
    ("#500724", "#831843", "#f43f5e"),  # Crimson Wine -> Magenta Velvet (Ruby accent)
    ("#0c4a6e", "#0369a1", "#38bdf8"),  # Steel Azure -> Cerulean (Ice Blue accent)
    ("#052e16", "#14532d", "#4ade80"),  # Dark Moss -> Evergreen (Emerald accent)
]


def wrap_text_into_lines(text: str, max_chars_per_line: int = 18, max_lines: int = 4) -> List[str]:
    """Wraps title text cleanly into lines respecting word boundaries."""
    words = text.split()
    lines = []
    current_words: List[str] = []
    current_length = 0

    for word in words:
        addition = len(word) + (1 if current_words else 0)
        if current_length + addition <= max_chars_per_line:
            current_words.append(word)
            current_length += addition
        else:
            if current_words:
                lines.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        if len(lines) >= max_lines:
            break

    if current_words and len(lines) < max_lines:
        lines.append(" ".join(current_words))

    # If title was longer than max_lines, truncate last line with ellipsis
    if len(lines) == max_lines and len(lines) < len(words):
        if not lines[-1].endswith("..."):
            lines[-1] = lines[-1].rstrip(".,:;") + "..."

    return lines


def generate_book_cover_svg(title: str, brand: str = "", subcategory: str = "Book") -> str:
    """
    Generates a deterministic SVG book cover with 3:4 aspect ratio (600x800).
    Uses SHA-256 of the title to deterministically choose the color palette.
    """
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
    palette_idx = int(digest[:8], 16) % len(PALETTES)
    c1, c2, accent = PALETTES[palette_idx]

    lines = wrap_text_into_lines(title, max_chars_per_line=19, max_lines=4)
    line_count = max(1, len(lines))

    # Dynamic font sizing based on line count and length
    if line_count <= 2:
        font_size = 40
        line_height = 50
        start_y = 280
    elif line_count == 3:
        font_size = 34
        line_height = 44
        start_y = 250
    else:
        font_size = 30
        line_height = 40
        start_y = 230

    title_elements = []
    for i, line in enumerate(lines):
        y = start_y + (i * line_height)
        safe_line = html.escape(line)
        title_elements.append(
            f'<text x="64" y="{y}" font-family="Georgia, \'Times New Roman\', serif" '
            f'font-size="{font_size}" font-weight="700" fill="#ffffff">{safe_line}</text>'
        )
    title_svg_block = "\n    ".join(title_elements)

    safe_brand = html.escape(brand or "SmartCart Publishing")
    safe_subcat = html.escape(subcategory or "Edition")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 800" width="600" height="800">
  <defs>
    <linearGradient id="cover_grad_{palette_idx}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{c1}" />
      <stop offset="100%" stop-color="{c2}" />
    </linearGradient>
    <linearGradient id="spine_grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#000000" stop-opacity="0.3" />
      <stop offset="50%" stop-color="#ffffff" stop-opacity="0.1" />
      <stop offset="100%" stop-color="#000000" stop-opacity="0.4" />
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.4" flood-color="#000000" />
    </filter>
  </defs>

  <!-- Background Base Gradient -->
  <rect width="600" height="800" fill="url(#cover_grad_{palette_idx})" />

  <!-- Book Spine Accents (Left Margin) -->
  <rect x="0" y="0" width="40" height="800" fill="url(#spine_grad)" />
  <line x1="40" y1="0" x2="40" y2="800" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.5" />

  <!-- Top Pill Tag: 'BOOK' -->
  <g transform="translate(64, 56)">
    <rect width="84" height="28" rx="6" fill="#ffffff" fill-opacity="0.15" stroke="#ffffff" stroke-opacity="0.25" stroke-width="1"/>
    <text x="42" y="19" font-family="system-ui, -apple-system, sans-serif" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle" letter-spacing="2">BOOK</text>
  </g>

  <!-- Subcategory Label -->
  <text x="64" y="180" font-family="system-ui, -apple-system, sans-serif" font-size="13" font-weight="700" fill="{accent}" letter-spacing="1.5" text-transform="uppercase">{safe_subcat.upper()}</text>

  <!-- Book Title (Serif, High Contrast) -->
  <g filter="url(#shadow)">
    {title_svg_block}
  </g>

  <!-- Divider Rule -->
  <line x1="64" y1="650" x2="164" y2="650" stroke="{accent}" stroke-width="2.5" stroke-linecap="round" />

  <!-- Publisher / Author Brand at Bottom -->
  <text x="64" y="705" font-family="system-ui, -apple-system, sans-serif" font-size="19" font-weight="600" fill="#f8fafc" letter-spacing="0.5">{safe_brand}</text>
  <text x="64" y="732" font-family="system-ui, -apple-system, sans-serif" font-size="13" font-weight="500" fill="#94a3b8">SmartCart Library Edition</text>
</svg>"""
    return svg


def get_book_cover_filename(title: str, product_id: int = None) -> str:
    """Generates a deterministic filename for a book cover."""
    slug_hash = hashlib.sha256(f"{title}_{product_id or ''}".encode("utf-8")).hexdigest()[:12]
    prefix = f"book_{product_id}_" if product_id is not None else "book_"
    return f"{prefix}{slug_hash}.svg"


def save_book_cover(title: str, brand: str = "", subcategory: str = "Book", product_id: int = None) -> str:
    """
    Saves an SVG book cover into backend/static/covers/ if not already present or changed.
    Returns the relative path: /static/covers/<filename>.svg
    """
    COVERS_DIR.mkdir(parents=True, exist_ok=True)
    filename = get_book_cover_filename(title, product_id)
    file_path = COVERS_DIR / filename
    svg_content = generate_book_cover_svg(title, brand, subcategory)

    # Idempotent write: write if missing or size/content changed
    if not file_path.exists() or file_path.read_text(encoding="utf-8") != svg_content:
        file_path.write_text(svg_content, encoding="utf-8")

    return f"/static/covers/{filename}"

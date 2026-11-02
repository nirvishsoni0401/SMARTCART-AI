import random
from datetime import datetime
from app.database import SessionLocal, init_db
from app.models import Product, User, CartItem, WishlistItem, Order, OrderItem, InteractionEvent, RecommendationLog, ModelState
from app.auth import hash_password

CATEGORIES = [
    "Electronics",
    "Fashion",
    "Home & Kitchen",
    "Beauty",
    "Sports",
    "Books"
]

BRANDS_BY_CAT = {
    "Electronics": ["boAt", "Noise", "OnePlus", "Samsung", "Xiaomi", "Realme", "Fire-Boltt", "Zebronics", "Sony", "Portronics"],
    "Fashion": ["FabIndia", "Peter England", "Allen Solly", "Manyavar", "Biba", "Bata", "Woodland", "Raymond", "Titan", "Fastrack"],
    "Home & Kitchen": ["Prestige", "Hawkins", "Pigeon", "Milton", "Wonderchef", "Butterfly", "Borosil", "Bajaj", "Cello", "Philips"],
    "Beauty": ["Mamaearth", "WOW Skin Science", "Plum", "Biotique", "Forest Essentials", "mCaffeine", "Dot & Key", "Lakme", "Minimalist", "Himalaya"],
    "Sports": ["Nivia", "Cosco", "Decathlon", "HRX", "Cultsport", "Boldfit", "Vector X", "Yonex", "SG", "Puma"],
    "Books": ["Penguin India", "HarperCollins", "Rupa Publications", "Jaico", "Westland", "O'Reilly", "Packt", "Bloomsbury", "S. Chand", "Pan Macmillan"]
}

TEMPLATES = {
    "Electronics": [
        ("{brand} Wave Pro Smartwatch with Bluetooth Calling & AMOLED Display", 1999, 4999, "Wearables"),
        ("{brand} Rockerz Wireless Noise Cancelling Over-Ear Headphones", 2499, 5999, "Audio"),
        ("{brand} BassHeads in-Ear Wired Earphones with HD Mic", 499, 999, "Audio"),
        ("{brand} Turbo 65W GaN Fast Charger Multi-Port Adapter", 1299, 2499, "Accessories"),
        ("{brand} Horizon 5G Smartphone (128GB, 8GB RAM, 50MP AI Camera)", 18999, 24999, "Mobile"),
        ("{brand} Smart Android TV Stick 4K with Voice Remote", 2999, 4499, "Home Entertainment"),
        ("{brand} Mechanical Gaming Keyboard with RGB Backlit Keys", 2199, 3999, "Computing"),
        ("{brand} Wireless Ergonomic Optical Mouse with Silent Click", 799, 1499, "Computing"),
        ("{brand} Ultra Slim 20000mAh Power Bank with Power Delivery", 1699, 3299, "Accessories"),
        ("{brand} True Wireless Stereo Earbuds with 40H Playtime", 1499, 3499, "Audio"),
        ("{brand} 1080p Full HD Web Camera with Privacy Shutter", 1599, 2999, "Computing"),
        ("{brand} Portable Waterproof Bluetooth Speaker with Extra Bass", 1899, 3999, "Audio")
    ],
    "Fashion": [
        ("{brand} Pure Cotton Slim Fit Casual Shirt", 1199, 2299, "Men's Clothing"),
        ("{brand} Classic Straight Fit Denim Jeans", 1599, 3199, "Men's Clothing"),
        ("{brand} Printed Anarkali Kurta with Palazzo & Dupatta", 1899, 3999, "Women's Clothing"),
        ("{brand} Genuine Leather Formal Dress Shoes for Men", 2499, 4999, "Footwear"),
        ("{brand} Breathable Mesh Running Athletic Sneakers", 1799, 3599, "Footwear"),
        ("{brand} Analog Dial Stainless Steel Watch with Date Function", 2299, 4599, "Watches"),
        ("{brand} Handcrafted Leather Bi-Fold Wallet with RFID Protection", 899, 1699, "Accessories"),
        ("{brand} Traditional Silk Blend Festive Saree with Blouse Piece", 2799, 5999, "Ethnic Wear"),
        ("{brand} Canvas Laptop Backpack with Anti-Theft Pocket", 1299, 2499, "Bags"),
        ("{brand} Polarized UV Protection Aviator Sunglasses", 999, 1999, "Eyewear"),
        ("{brand} Linen Blend Mandarin Collar Kurta", 1499, 2799, "Ethnic Wear"),
        ("{brand} Structured Vegan Leather Shoulder Tote Handbag", 1999, 3799, "Bags")
    ],
    "Home & Kitchen": [
        ("{brand} Hard Anodised Non-Stick Induction Base Frying Pan", 1099, 1999, "Cookware"),
        ("{brand} Contura Aluminium Outer Lid Pressure Cooker 3L", 1599, 2499, "Cookware"),
        ("{brand} 750W Heavy Duty Mixer Grinder with 3 Stainless Steel Jars", 2999, 4999, "Appliances"),
        ("{brand} Double Wall Vacuum Insulated Stainless Steel Flask 1000ml", 899, 1499, "Storage"),
        ("{brand} Microfibre Reversible All-Season AC Comforter Double Bed", 1799, 3299, "Bedding"),
        ("{brand} 100% Pure Egyptian Cotton 300 TC King Bed Sheet Set", 1499, 2799, "Bedding"),
        ("{brand} Elegant Ceramic Dinnerware Set 18 Pieces", 2499, 4599, "Tableware"),
        ("{brand} Automatic Electric Kettle 1.8 Litre Fast Boiling", 799, 1499, "Appliances"),
        ("{brand} Modular Airtight Kitchen Food Storage Containers (Set of 6)", 849, 1599, "Storage"),
        ("{brand} Aroma Essential Oil Diffuser with 7 LED Ambient Colors", 1199, 2199, "Home Decor"),
        ("{brand} Memory Foam Ergonomic Cervical Sleeping Pillow", 1299, 2599, "Bedding"),
        ("{brand} Multipurpose Engineered Wood 3-Tier Wall Shelf Rack", 999, 1899, "Home Decor")
    ],
    "Beauty": [
        ("{brand} Vitamin C Daily Glow Face Serum for Brightening (30ml)", 549, 999, "Skincare"),
        ("{brand} Onion Anti-Hair Fall Shampoo with Plant Keratin (400ml)", 429, 699, "Haircare"),
        ("{brand} Green Tea Pore Cleansing Face Wash with Glycolic Acid", 349, 499, "Skincare"),
        ("{brand} Ultra Matte Long-Lasting Weightless Liquid Lipstick", 499, 799, "Makeup"),
        ("{brand} Naked & Raw Coffee Face Scrub for Tan Removal (100g)", 399, 599, "Skincare"),
        ("{brand} 10% Niacinamide Daily Clarifying Serum with Zinc", 599, 899, "Skincare"),
        ("{brand} Hyaluronic Acid Dewy Sunscreen Gel SPF 50+ PA++++", 499, 749, "Sun Care"),
        ("{brand} Pure Cold Pressed Bhringraj Hair Oil for Nourishment", 379, 599, "Haircare"),
        ("{brand} Refreshing Eau De Parfum Long Lasting Luxury Fragrance (100ml)", 1199, 2199, "Fragrances"),
        ("{brand} Deep Moisturizing Cocoa Butter Body Lotion with Shea (400ml)", 399, 649, "Body Care"),
        ("{brand} Tea Tree & Salicylic Acid Clarifying Acne Spot Gel", 299, 499, "Skincare"),
        ("{brand} All-in-One Beard Grooming Oil & Shaving Balm Kit", 699, 1199, "Men's Grooming")
    ],
    "Sports": [
        ("{brand} Storm Rubber Moulded Football Size 5 Outdoor Match Ball", 749, 1299, "Football"),
        ("{brand} Professional English Willow Cricket Bat Full Size", 4499, 7999, "Cricket"),
        ("{brand} Anti-Slip 6mm High Density Yoga Mat with Carrying Strap", 799, 1499, "Fitness & Yoga"),
        ("{brand} Solid Hexagonal Rubber Coated Dumbbells (Pair 5kg)", 1599, 2699, "Gym Equipment"),
        ("{brand} High Speed Adjustable Bearing Skipping Jump Rope", 299, 599, "Cardio"),
        ("{brand} Muscle Recovery Deep Tissue Massage Gun 6 Speeds", 2499, 4999, "Recovery"),
        ("{brand} Men's Dry-Fit Moisture Wicking Training Gym T-Shirt", 599, 1199, "Activewear"),
        ("{brand} Graphite Badminton Racket with High Tension Guts", 1699, 3199, "Badminton"),
        ("{brand} Heavy Duty Resistance Bands Set (5 Tube Loops)", 899, 1699, "Gym Equipment"),
        ("{brand} Thermal Cycling Water Bottle with Dust Cap (750ml)", 399, 699, "Cycling"),
        ("{brand} Wrist Support Gym Gloves with Non-Slip Grip", 449, 799, "Accessories"),
        ("{brand} Heavy Duty 40kg Home Gym Fitness Set with Rods", 3499, 5999, "Gym Equipment")
    ],
    "Books": [
        ("Grokking Machine Learning & Bandit Algorithms", 899, 1299, "Computer Science"),
        ("Clean Code: Agile Software Craftsmanship", 799, 1199, "Programming"),
        ("Designing Data-Intensive Applications", 1299, 1899, "Technology"),
        ("Atomic Habits: Easy & Proven Way to Build Good Habits", 499, 799, "Self-Help"),
        ("The Psychology of Money: Timeless Lessons on Wealth", 399, 599, "Business"),
        ("Ikigai: The Japanese Secret to a Long and Happy Life", 349, 499, "Philosophy"),
        ("System Design Interview: An Insider's Guide", 1499, 2199, "Technology"),
        ("Hands-On Reinforcement Learning with Python", 1199, 1699, "Data Science"),
        ("Rich Dad Poor Dad: What the Rich Teach Their Kids", 399, 599, "Finance"),
        ("The Lean Startup: Continuous Innovation for Radical Success", 599, 899, "Business"),
        ("Sapiens: A Brief History of Humankind", 499, 799, "History"),
        ("Deep Work: Rules for Focused Success in a Distracted World", 449, 699, "Productivity")
    ]
}

def seed_database():
    print("[Seed] Initializing database tables...")
    init_db()
    db = SessionLocal()

    # Clear existing products to ensure clean seed
    existing_product_count = db.query(Product).count()
    if existing_product_count >= 600:
        print(f"[Seed] Products already seeded ({existing_product_count} products found). Skipping product seed.")
    else:
        print(f"[Seed] Clearing old product data (found {existing_product_count})...")
        db.query(CartItem).delete()
        db.query(WishlistItem).delete()
        db.query(OrderItem).delete()
        db.query(Order).delete()
        db.query(InteractionEvent).delete()
        db.query(RecommendationLog).delete()
        db.query(Product).delete()
        db.commit()

        print("[Seed] Generating 600+ realistic products across 6 departments...")
        random.seed(42)
        products = []
        product_id = 1

        # We generate 105 products per category => 630 products total
        for cat in CATEGORIES:
            brands = BRANDS_BY_CAT[cat]
            templates = TEMPLATES[cat]
            
            for i in range(105):
                brand = brands[i % len(brands)]
                tmpl_name, base_price, base_mrp, subcat = templates[i % len(templates)]
                
                # Add slight variance to prices and names
                price_mult = round(random.uniform(0.85, 1.35), 2)
                price = round(base_price * price_mult)
                mrp = round(max(price * 1.15, base_mrp * price_mult))
                discount_pct = round(((mrp - price) / mrp) * 100, 1)

                name = tmpl_name.replace("{brand}", brand)
                if i >= len(templates):
                    variant_suffixes = ["V2", "Pro", "Max", "Special Edition", "Signature Edition", "2026 Edition", "Plus", "Ultra"]
                    name = f"{name} - {variant_suffixes[i % len(variant_suffixes)]}"

                rating = round(random.uniform(3.7, 4.9), 1)
                rating_count = random.randint(35, 4800)
                popularity = round(random.uniform(40.0, 98.0), 1)
                stock = random.randint(15, 350)
                
                # Seeded image URL using picsum photos with deterministic seed
                img_seed = 1000 + product_id
                image_url = f"https://picsum.photos/seed/smartcart_{img_seed}/600/600"

                attributes = {
                    "Subcategory": subcat,
                    "Warranty": f"{random.choice([6, 12, 24])} Months Manufacturer Warranty",
                    "Country of Origin": "India",
                    "Return Policy": "7 Days Replacement / Return",
                    "Color": random.choice(["Black", "Navy Blue", "Charcoal Gray", "Classic White", "Silver", "Olive Green", "Teal"]),
                    "Seller": "SmartCart Verified Retailer"
                }

                desc = f"Experience top tier quality with the {name}. Designed with premium materials for maximum durability and satisfaction. Genuine {brand} product with authentic warranty and express delivery across India."

                product = Product(
                    id=product_id,
                    name=name,
                    category=cat,
                    brand=brand,
                    price=float(price),
                    mrp=float(mrp),
                    discount_pct=float(discount_pct),
                    rating=float(rating),
                    rating_count=rating_count,
                    popularity_score=float(popularity),
                    description=desc,
                    image_url=image_url,
                    attributes=attributes,
                    stock=stock,
                    created_at=datetime.utcnow()
                )
                products.append(product)
                product_id += 1

        db.add_all(products)
        db.commit()
        print(f"[Seed] Successfully seeded {len(products)} products across {len(CATEGORIES)} categories!")

    # Seed Demo Users and Admin
    demo_users = [
        {"name": "Admin User", "email": "admin@smartcart.ai", "password": "admin123", "role": "admin", "algo": "linucb"},
        {"name": "Rahul Sharma", "email": "rahul@smartcart.ai", "password": "user123", "role": "user", "algo": "linucb"},
        {"name": "Priya Patel", "email": "priya@smartcart.ai", "password": "user123", "role": "user", "algo": "discounted_linucb"},
        {"name": "Amit Kumar", "email": "amit@smartcart.ai", "password": "user123", "role": "user", "algo": "linucb_thompson"}
    ]

    for u_info in demo_users:
        user = db.query(User).filter(User.email == u_info["email"]).first()
        if not user:
            user = User(
                name=u_info["name"],
                email=u_info["email"],
                password_hash=hash_password(u_info["password"]),
                role=u_info["role"],
                assigned_algorithm=u_info["algo"],
                consent_tracking=True
            )
            db.add(user)
            print(f"[Seed] Created demo user: {u_info['email']} (role: {u_info['role']})")

    db.commit()
    db.close()
    print("[Seed] Seeding completed successfully.")

if __name__ == "__main__":
    seed_database()


"""
SmartCart AI - Large Scale Product Seeding Engine (10,000+ Products)
Generates high-quality, realistic, varied e-commerce product records in INR (₹)
across 10 major categories and 55 subcategories.
"""

import os
import sys
import random
import datetime
from pathlib import Path
from pymongo import MongoClient, TEXT
from dotenv import load_dotenv

# Ensure backend root is on sys.path
backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.database import (
    get_mongo_client,
    products_collection,
    init_db_indexes,
    DATABASE_NAME
)

# 10 Major Categories & 55 Subcategories Definition
TAXONOMY = {
    "Electronics": {
        "slug": "electronics",
        "subcategories": [
            "Smartphones", "Laptops", "Tablets", "Smartwatches", "Headphones",
            "Bluetooth Speakers", "Cameras", "Computer Accessories", "Gaming Consoles", "Monitors"
        ],
        "brands": [
            "Apple", "Samsung", "OnePlus", "Sony", "Dell", "HP", "Lenovo", "Asus",
            "boAt", "Noise", "JBL", "Xiaomi", "Realme", "Acer", "LG", "Canon",
            "Nikon", "Logitech", "Razer", "Zebronics"
        ],
        "image_pool": [
            "photo-1511707171634-5f897ff02aa9", "photo-1592750475338-74b7b21085ab",
            "photo-1598327105666-5b89351aff97", "photo-1580910051074-3eb694886505",
            "photo-1496181133206-80ce9b88a853", "photo-1517336714731-489689fd1ca8",
            "photo-1525547719571-a2d4ac8945e2", "photo-1505740420928-5e560c06d30e",
            "photo-1546435770-a3e426bf472b", "photo-1523275335684-37898b6baf30",
            "photo-1508685096489-7aacd43bd3b1", "photo-1527443224154-c4a3942d3acf"
        ],
        "price_range": (1499, 129999),
        "spec_templates": {
            "Smartphones": {"Display": ["6.7-inch AMOLED 120Hz", "6.1-inch Super Retina", "6.5-inch FHD+ 90Hz"], "Processor": ["Snapdragon 8 Gen 3", "Apple A17 Pro", "Dimensity 9300", "Snapdragon 7s Gen 2"], "RAM": ["8GB", "12GB", "16GB"], "Storage": ["128GB", "256GB", "512GB"], "Battery": ["5000 mAh 67W Turbo", "4500 mAh Fast Charge", "5400 mAh"]},
            "Laptops": {"Processor": ["Intel Core i7 14th Gen", "AMD Ryzen 7 7840HS", "Apple M3 Chip", "Intel Core i5 13th Gen"], "RAM": ["16GB DDR5", "32GB DDR5", "8GB DDR4"], "SSD": ["512GB NVMe M.2", "1TB Gen4 SSD", "2TB SSD"], "GPU": ["NVIDIA RTX 4060 8GB", "Intel Iris Xe", "NVIDIA RTX 3050 6GB"], "Display": ["15.6-inch 144Hz FHD", "14-inch OLED 2.8K", "16-inch 165Hz WQXGA"]},
            "Headphones": {"Type": ["Wireless Over-Ear ANC", "True Wireless In-Ear", "Gaming Headset 7.1"], "Battery Life": ["Up to 40 Hours", "Up to 50 Hours ANC", "30 Hours"], "Connectivity": ["Bluetooth 5.3 + AUX", "Bluetooth 5.4 LDAC", "Low Latency 2.4GHz"]},
            "Smartwatches": {"Display": ["1.43-inch AMOLED", "1.96-inch HD Always-On", "Retina OLED 1000 nits"], "Sensors": ["SpO2, Heart Rate, Sleep Tracker", "ECG, Blood Pressure, Temp", "Optical HR + GPS"], "Water Resistance": ["IP68 50m Water Resistant", "5ATM Water Resistance"]},
            "Default": {"Connectivity": ["Wireless", "USB-C", "Bluetooth 5.3"], "Warranty": ["1 Year Manufacturer Warranty", "2 Years Comprehensive Warranty"]}
        }
    },
    "Fashion": {
        "slug": "fashion",
        "subcategories": [
            "Men's Clothing", "Women's Clothing", "Kids' Clothing", "Shoes",
            "Watches", "Handbags", "Accessories"
        ],
        "brands": [
            "Nike", "Adidas", "Puma", "Levi's", "Zara", "H&M", "Allen Solly",
            "Peter England", "Van Heusen", "Biba", "FabIndia", "Fossil",
            "Titan", "Fastrack", "Lavie", "Bata", "Woodland", "Red Tape"
        ],
        "image_pool": [
            "photo-1521572267360-ee0c2909d518", "photo-1602810318383-e386cc2a3ccf",
            "photo-1490481651871-ab68de25d43d", "photo-1515886657613-9f3515b0c78f",
            "photo-1542291026-7eec264c27ff", "photo-1595950653106-6c9ebd614d3a",
            "photo-1523275335684-37898b6baf30", "photo-1584917865442-de89df76afd3",
            "photo-1548036328-c9fa89d128fa", "photo-1600185365926-3a2ce3cdb9eb"
        ],
        "price_range": (399, 14999),
        "spec_templates": {
            "Shoes": {"Sole Material": ["EVA Cushion", "Rubber Non-slip", "Memory Foam"], "Upper Material": ["Breathable Mesh", "Genuine Leather", "Synthetic Canvas"], "Closure": ["Lace-Up", "Slip-On", "Velcro"]},
            "Watches": {"Movement": ["Japanese Quartz", "Automatic Self-Wind", "Chronograph Multi-Function"], "Dial Glass": ["Mineral Crystal", "Sapphire Glass"], "Strap Material": ["Stainless Steel", "Genuine Leather", "Silicone"]},
            "Default": {"Fabric": ["100% Combed Cotton", "Linen Blend Breathable", "Premium Polyester", "Denim Stretch"], "Fit": ["Regular Fit", "Slim Fit", "Relaxed Fit"], "Care": ["Machine Wash Cold", "Dry Clean Recommended", "Hand Wash Only"]}
        }
    },
    "Home & Kitchen": {
        "slug": "home-and-kitchen",
        "subcategories": [
            "Furniture", "Kitchen Appliances", "Home Decor", "Lighting", "Storage Products", "Cookware"
        ],
        "brands": [
            "Prestige", "Hawkins", "Philips", "Bajaj", "Wonderchef", "Godrej",
            "IKEA", "Sleepwell", "Wakefit", "Crompton", "Milton", "Borosil", "Cello"
        ],
        "image_pool": [
            "photo-1556911220-e15b29be8c8f", "photo-1585659722983-3a675dabf23d",
            "photo-1590794056226-79ef3a8147e1", "photo-1513519245088-0e12902e5a38",
            "photo-1534349762230-e0cadf78f5da", "photo-1586023492125-27b2c045efd7",
            "photo-1507473885765-e6ed057f782c", "photo-1517705008128-361805f42e86"
        ],
        "price_range": (299, 45999),
        "spec_templates": {
            "Cookware": {"Material": ["Triply Stainless Steel", "Hard Anodized Aluminum", "Cast Iron Pre-Seasoned"], "Induction Compatible": ["Yes", "Both Gas and Induction", "Gas Stove Only"], "Coating": ["Granite Non-Stick", "Food-Grade Stainless"]},
            "Kitchen Appliances": {"Power Wattage": ["750W Heavy Duty", "1000W Inverter Motor", "1200W High Speed"], "Voltage": ["230V 50Hz", "220-240V AC"], "Warranty": ["2 Years Motor Warranty", "5 Years Extended Warranty"]},
            "Default": {"Material": ["Engineered Wood", "High-Grade Stainless Steel", "Virgin BPA-Free Plastic"], "Assembly": ["Pre-assembled", "DIY with Tool Kit", "Carpenter Assembly Required"]}
        }
    },
    "Beauty & Personal Care": {
        "slug": "beauty-and-personal-care",
        "subcategories": [
            "Skincare", "Haircare", "Makeup", "Grooming Products", "Personal Care Appliances"
        ],
        "brands": [
            "L'Oreal", "Maybelline", "Lakme", "Nivea", "Neutrogena", "The Ordinary",
            "Mamaearth", "Biotique", "Philips", "Braun", "Gillette", "Beardo", "Minimalist"
        ],
        "image_pool": [
            "photo-1556228720-195a672e8a03", "photo-1571781926291-c477ebfd024b",
            "photo-1522337360788-8b13dee7a37e", "photo-1598440947619-2c35fc9aa908",
            "photo-1512290900672-1f48f4305ff1", "photo-1527799820374-dcf8d9d4a388"
        ],
        "price_range": (199, 7999),
        "spec_templates": {
            "Personal Care Appliances": {"Blade Material": ["Self-Sharpening Stainless Steel", "Titanium Coated"], "Battery Life": ["90 Mins Cordless", "120 Mins Quick Charge"], "Waterproof": ["IPX7 100% Washable", "Splash-Proof"]},
            "Default": {"Skin / Hair Type": ["All Skin Types", "Sensitive & Acne Prone", "Normal to Dry", "Oily to Combination"], "Ingredients": ["Hyaluronic Acid 2% + B5", "Niacinamide 10% + Zinc", "Salicylic Acid", "Vitamin C 15%"], "Cruelty Free": ["Yes, Dermatologically Tested", "100% Vegan & Paraben-Free"]}
        }
    },
    "Sports & Fitness": {
        "slug": "sports-and-fitness",
        "subcategories": [
            "Gym Equipment", "Sportswear", "Fitness Trackers", "Outdoor Equipment", "Cycling Accessories"
        ],
        "brands": [
            "Decathlon", "Nivia", "Yonex", "Cosco", "Kore", "Fitkit", "Cultsport",
            "Strauss", "Boldfit", "Vector X", "Firefox", "Garmin"
        ],
        "image_pool": [
            "photo-1584735935682-2f2b69dff9d2", "photo-1517838277536-f5f99be501cd",
            "photo-1534438327276-14e5300c3a48", "photo-1574680096145-d05b474e2155",
            "photo-1517649763962-0c623266ddc0", "photo-1486739985386-d4fae04ca6f7"
        ],
        "price_range": (299, 39999),
        "spec_templates": {
            "Gym Equipment": {"Material": ["Solid Cast Iron", "PVC Rubber Encased", "High-Tensile Steel Frame"], "Max Weight Capacity": ["150 kg", "200 kg", "120 kg"], "Included Accessories": ["Safety Lock Collars", "Exercise Manual", "Resistance Bands"]},
            "Default": {"Sport Type": ["Fitness & Gym", "Running & Marathon", "Badminton & Tennis", "Outdoor Trekking"], "Features": ["Sweat Wicking Quick Dry", "Shock Absorbing High Durability", "Ergonomic Grip"]}
        }
    },
    "Books": {
        "slug": "books",
        "subcategories": [
            "Programming", "Technology", "Fiction", "Business", "Educational Books", "Self-Improvement"
        ],
        "brands": [
            "O'Reilly", "Penguin Random House", "HarperCollins", "Pearson", "Packt",
            "Rupa Publications", "Bloomsbury", "McGraw Hill", "Wiley"
        ],
        "image_pool": [
            "photo-1544716278-ca5e3f4abd8c", "photo-1532012164546-f432f2e3777a",
            "photo-1512820790803-83ca734da794", "photo-1497633762265-9d179a990aa6",
            "photo-1495446815901-a7297e633e8d", "photo-1457369804613-52c61a468e7d"
        ],
        "price_range": (199, 4999),
        "spec_templates": {
            "Default": {"Format": ["Paperback", "Hardcover", "Collector's Edition"], "Language": ["English", "Hindi"], "Page Count": ["320 Pages", "480 Pages", "260 Pages", "650 Pages"], "Publisher": ["Global Editions", "Authorized Academic Press"]}
        }
    },
    "Automotive": {
        "slug": "automotive",
        "subcategories": [
            "Car Accessories", "Bike Accessories", "Vehicle Electronics", "Maintenance Products"
        ],
        "brands": [
            "Bosch", "3M", "Motul", "Michelin", "Pioneer", "Amaron", "Castrol",
            "Vega", "Steelbird", "Studds", "TVS", "Formula 1"
        ],
        "image_pool": [
            "photo-1503376780353-7e6692767b70", "photo-1511919884226-fd3cad34687c",
            "photo-1489824904134-891ab64532f1", "photo-1563720223185-11003d516935",
            "photo-1508974239320-0a029497e820", "photo-1549399542-7e3f8b79c341"
        ],
        "price_range": (249, 24999),
        "spec_templates": {
            "Default": {"Compatibility": ["Universal Fit All Cars", "Universal Fit All Motorcycles", "SUV & Sedan"], "Material": ["Industrial High Grade ABS", "Reinforced Composite", "Synthetic Waterproof"], "Warranty": ["1 Year Replacement Guarantee", "6 Months Standard"]}
        }
    },
    "Toys & Baby Products": {
        "slug": "toys-and-baby",
        "subcategories": [
            "Educational Toys", "Board Games", "Baby Care", "Kids' Toys"
        ],
        "brands": [
            "Lego", "Hasbro", "Mattel", "Hot Wheels", "Fisher-Price", "Chicco",
            "Himalaya Baby", "Pampers", "MamyPoko", "Funskool", "Nerf"
        ],
        "image_pool": [
            "photo-1566576912321-d58ddd7a6088", "photo-1558060370-d644479cb6f7",
            "photo-1596461404969-9ae70f2830c1", "photo-1587654780291-39c9404d746b",
            "photo-1515488042361-ee00e0ddd4e4", "photo-1500995617113-cf789362a3e1"
        ],
        "price_range": (199, 12999),
        "spec_templates": {
            "Default": {"Recommended Age": ["0-2 Years", "3-6 Years", "6-12 Years", "8+ Years Family"], "Safety Standard": ["BIS Certified Non-Toxic", "100% BPA-Free & Safe", "EN71 Certified"], "Battery Required": ["No Batteries Required", "2x AA Batteries (Included)", "USB Rechargeable"]}
        }
    },
    "Grocery & Food": {
        "slug": "grocery-and-food",
        "subcategories": [
            "Snacks", "Beverages", "Packaged Food", "Household Essentials"
        ],
        "brands": [
            "Tata Tea", "Nestle", "Cadbury", "Britannia", "Amul", "Haldiram's",
            "Fortune", "Dettol", "Surf Excel", "Colgate", "Real", "Lipton", "Sunfeast"
        ],
        "image_pool": [
            "photo-1542838132-92c53300491e", "photo-1579113800032-c38bd7635818",
            "photo-1607623814075-e51df1bdc82f", "photo-1528735602780-2552fd46c7af",
            "photo-1563636619-e9143da7973b", "photo-1509440159596-0249088772ff"
        ],
        "price_range": (49, 2499),
        "spec_templates": {
            "Default": {"Dietary Preference": ["100% Vegetarian (Green Dot)", "Gluten Free", "Zero Preservatives"], "Net Quantity": ["500g Pack", "1kg Value Pack", "Pack of 3 (3x200g)", "750ml Bottle"], "Shelf Life": ["9 Months from Packaging", "12 Months from MFD", "6 Months Best Before"]}
        }
    },
    "Office & Stationery": {
        "slug": "office-and-stationery",
        "subcategories": [
            "Office Supplies", "Notebooks", "Printers", "Desk Accessories"
        ],
        "brands": [
            "Classmate", "Reynolds", "Parker", "Camlin", "Faber-Castell", "HP",
            "Canon", "Epson", "Kangaro", "Solo", "Fellowes"
        ],
        "image_pool": [
            "photo-1585776245991-cf89dd7fc73a", "photo-1583485088034-697b5bc54ccd",
            "photo-1517842645767-c639042777db", "photo-1586075010923-2dd4570fb338",
            "photo-1507842229451-7f01be8610ce", "photo-1544816155-12df9643f363"
        ],
        "price_range": (99, 32999),
        "spec_templates": {
            "Printers": {"Print Technology": ["All-in-One Wireless Ink Tank", "Laser Monochrome High Speed", "Photo Inkjet Duplex"], "Connectivity": ["Wi-Fi, USB 2.0, Mobile App", "Ethernet + Wi-Fi Direct"], "Yield": ["Up to 7,500 Color Pages", "Up to 5,000 Mono Pages"]},
            "Default": {"Material": ["Acid-Free 80 GSM Paper", "Aerospace Grade Aluminum", "Hardbound Vegan Leather"], "Package Contents": ["1 Unit Desk Organizer", "Pack of 6 Premium Pens", "1 Notebook 240 Pages Single Ruled"]}
        }
    }
}

SELLERS = [
    "RetailNet India Pvt Ltd",
    "Appario Retail Solutions",
    "Cloudtail Lifestyle Commerce",
    "OmniTech SuperStore",
    "TrueCom Enterprises",
    "IndoTrade Direct",
    "PrimeKart Logistics",
    "MetroCommerce Hub"
]

PRODUCT_ADJECTIVES = [
    "Pro", "Max", "Ultra", "Elite", "Prime", "Classic", "Smart", "Compact",
    "Heavy-Duty", "Premium", "Wireless", "Ergonomic", "Advanced", "Signature",
    "Precision", "Turbo", "Eco-Friendly", "Deluxe", "Next-Gen", "High-Performance"
]

COLOR_VARIANTS = [
    "Midnight Black", "Arctic White", "Space Gray", "Navy Blue", "Forest Green",
    "Champagne Gold", "Crimson Red", "Rose Gold", "Titanium Silver", "Deep Purple"
]


def build_image_url(photo_id: str, seed: int = 1) -> str:
    """Build high quality, fast-loading Unsplash CDN image URL with auto-crop"""
    return f"https://images.unsplash.com/{photo_id}?auto=format&fit=crop&w=800&q=80&sig={seed}"


def generate_single_product(cat_name: str, subcat_name: str, cat_info: dict, index: int, total_in_subcat: int) -> dict:
    """Generate a single realistic product record"""
    brand = cat_info["brands"][index % len(cat_info["brands"])]
    adj = PRODUCT_ADJECTIVES[(index * 7) % len(PRODUCT_ADJECTIVES)]
    color = COLOR_VARIANTS[(index * 3) % len(COLOR_VARIANTS)]
    series_num = (index % 50) + 1
    
    # Meaningful title generation by subcategory
    title_pattern = f"{brand} {adj} {subcat_name.rstrip('s')} {series_num} Series ({color})"
    if subcat_name == "Smartphones":
        title_pattern = f"{brand} {adj} Phone {series_num} 5G ({color}, {(index % 4 + 1) * 64}GB)"
    elif subcat_name == "Laptops":
        title_pattern = f"{brand} {adj}Book {series_num} Pro Laptop 15.6\" ({color})"
    elif subcat_name == "Headphones":
        title_pattern = f"{brand} {adj}Sound ANC Wireless Headphones ({color})"
    elif subcat_name == "Watches":
        title_pattern = f"{brand} {adj} Analog-Digital Multi-Dial Watch ({color})"
    elif subcat_name == "Shoes":
        title_pattern = f"{brand} {adj} Running & Training Shoes ({color}, UK Size {(index % 5) + 7})"
    elif subcat_name == "Programming":
        techs = ["Python", "JavaScript", "Rust", "Go", "Kubernetes", "AI & ML", "FastAPI", "React 19", "System Design", "Distributed Systems"]
        t = techs[index % len(techs)]
        title_pattern = f"Mastering {t}: {adj} Guide for Production Systems by {brand}"
    elif subcat_name == "Snacks":
        title_pattern = f"{brand} {adj} Crispy Roasted Crunch Snacks ({color.split()[0]} Edition, 400g)"

    # Price and discount
    min_p, max_p = cat_info["price_range"]
    base_mrp = random.randint(int(min_p * 1.2), int(max_p))
    discount_pct = random.choice([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    discounted_price = round(base_mrp * (1 - discount_pct / 100.0), 2)

    # Realistic rating & review counts
    rating = round(random.uniform(3.4, 4.9), 1)
    review_count = int(random.triangular(15, 4800, 350))
    popularity_score = round(min(99.9, (rating * 15.0) + (discount_pct * 0.3) + min(15.0, review_count / 200.0)), 1)

    # Stock & availability
    stock_quantity = random.randint(3, 450)
    availability = "In Stock" if stock_quantity > 10 else ("Low Stock" if stock_quantity > 0 else "Out of Stock")

    # Images: pick 3 distinct images from category pool
    pool = cat_info["image_pool"]
    img1 = build_image_url(pool[index % len(pool)], seed=index)
    img2 = build_image_url(pool[(index + 1) % len(pool)], seed=index + 100)
    img3 = build_image_url(pool[(index + 2) % len(pool)], seed=index + 200)
    images = [img1, img2, img3]

    # Specifications
    spec_dict = {}
    spec_templates = cat_info.get("spec_templates", {})
    subcat_spec = spec_templates.get(subcat_name, spec_templates.get("Default", {}))
    for spec_k, spec_vals in subcat_spec.items():
        spec_dict[spec_k] = spec_vals[index % len(spec_vals)]
    spec_dict["Brand"] = brand
    spec_dict["Color"] = color
    spec_dict["Model Number"] = f"{brand[:3].upper()}-{series_num:03d}-{adj[:3].upper()}"

    # Search Tags
    tags = [
        brand.lower(),
        cat_name.lower(),
        subcat_name.lower(),
        adj.lower(),
        color.lower(),
        f"best {subcat_name.lower()}",
        f"{brand.lower()} {subcat_name.lower()}"
    ]

    # Meaningful description
    description = (
        f"Experience superior quality and modern performance with the all-new {title_pattern}. "
        f"Crafted by {brand} with high-precision engineering and premium materials, this {subcat_name.lower()} "
        f"delivers outstanding reliability, elegant design in {color}, and exceptional value. "
        f"Backed by official seller warranty from {SELLERS[index % len(SELLERS)]}."
    )

    product_id = 10000 + index
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "product_id": product_id,
        "external_id": product_id, # Backward compatibility with existing bandit & cart
        "title": title_pattern,
        "name": title_pattern,     # Backward compatibility with existing listing & search
        "description": description,
        "brand": brand,
        "category": cat_name,
        "category_slug": cat_info["slug"],
        "subcategory": subcat_name,
        "subcategory_slug": subcat_name.lower().replace(" ", "-").replace("'", "").replace("&", "and"),
        "price": float(base_mrp),
        "discounted_price": float(discounted_price),
        "discount_percentage": discount_pct,
        "currency": "INR",
        "rating": rating,
        "review_count": review_count,
        "stock_quantity": stock_quantity,
        "stock": stock_quantity,   # Backward compatibility with existing endpoints
        "availability": availability,
        "images": images,
        "thumbnail": images[0],
        "image_url": images[0],    # Backward compatibility with existing UI
        "seller": SELLERS[index % len(SELLERS)],
        "specifications": spec_dict,
        "tags": tags,
        "popularity_score": popularity_score,
        "is_active": True,
        "created_at": now_iso,
        "updated_at": now_iso
    }


def seed_database(target_total: int = 10500, batch_size: int = 1000, drop_first: bool = True):
    """
    Main seeding routine: generates realistic product records and bulk inserts them.
    """
    client = get_mongo_client()
    db = client[DATABASE_NAME]
    col = db["products"]

    print("=================================================================")
    print("       SMARTCART AI - LARGE SCALE PRODUCT CATALOG SEEDER         ")
    print("=================================================================")
    print(f"Target Products:  {target_total:,}")
    print(f"Target Database:  {DATABASE_NAME}")
    print(f"Target Collection: products")
    print(f"Categories:       {len(TAXONOMY)}")
    total_subcategories = sum(len(c["subcategories"]) for c in TAXONOMY.values())
    print(f"Subcategories:    {total_subcategories}")
    print("=================================================================")

    if drop_first:
        print("[Seeder] Clearing existing products collection...")
        col.delete_many({})

    # Calculate count per subcategory to reach target_total
    count_per_subcat = int(target_total / total_subcategories) + 1
    total_products_to_gen = count_per_subcat * total_subcategories

    print(f"[Seeder] Generating ~{count_per_subcat} products per subcategory (Total: {total_products_to_gen:,})...")

    batch = []
    global_index = 1
    inserted_total = 0

    for cat_name, cat_info in TAXONOMY.items():
        print(f"\n-> Generating category '{cat_name}' ({len(cat_info['subcategories'])} subcategories)...")
        for subcat in cat_info["subcategories"]:
            for i in range(count_per_subcat):
                product_doc = generate_single_product(
                    cat_name=cat_name,
                    subcat_name=subcat,
                    cat_info=cat_info,
                    index=global_index,
                    total_in_subcat=count_per_subcat
                )
                batch.append(product_doc)
                global_index += 1

                if len(batch) >= batch_size:
                    col.insert_many(batch, ordered=False)
                    inserted_total += len(batch)
                    print(f"   [Inserted] {inserted_total:,} / {total_products_to_gen:,} products...")
                    batch = []

    # Final batch
    if batch:
        col.insert_many(batch, ordered=False)
        inserted_total += len(batch)
        print(f"   [Inserted] {inserted_total:,} / {total_products_to_gen:,} products...")
        batch = []

    print(f"\n[Seeder] Product insertion complete! Total in collection: {col.count_documents({}):,}")
    print("[Seeder] Initializing database indexes...")
    init_db_indexes()
    print("=================================================================")
    print("   PRODUCT CATALOG SEEDING COMPLETED SUCCESSFULLY!              ")
    print("=================================================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed SmartCart AI product catalog")
    parser.add_argument("--count", type=int, default=10450, help="Target total products (default: 10450)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Insert batch size (default: 1000)")
    parser.add_argument("--keep-existing", action="store_true", help="Do not drop existing products")
    args = parser.parse_args()

    seed_database(
        target_total=args.count,
        batch_size=args.batch_size,
        drop_first=not args.keep_existing
    )


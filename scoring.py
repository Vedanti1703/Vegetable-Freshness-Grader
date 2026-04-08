"""
scoring.py — Freshness Score, Grade, Shelf Life & Price Calculation
===================================================================
All scoring logic is centralized here, fully decoupled from UI and prediction.

Freshness Score Formula:
  1. ML base score: maps prediction + confidence to a 0–100 scale
  2. CV quality score: derived from saturation, rot %, and wilting %
  3. Final = 70% ML + 30% CV  (ML is the primary signal)

Grade Scale:
  A (85–100) → Very Fresh
  B (60–84)  → Fresh
  C (40–59)  → Aging / Use Soon
  D (0–39)   → Rotten / Avoid
"""

# ══════════════════════════════════════════════════════════════════
# VEGETABLE DATABASE — Prices & Shelf Life (Indian Market)
# ══════════════════════════════════════════════════════════════════

# Base shelf life in days when perfectly fresh (Grade A)
BASE_SHELF_LIFE = {
    "Potato": 14, "Tomato": 7, "Onion": 21, "Carrot": 12,
    "Cabbage": 10, "Bell Pepper": 7, "Broccoli": 5, "Cucumber": 5,
    "Apple": 14, "Banana": 5, "Orange": 14, "Lettuce": 4,
    "Capsicum": 7, "Spinach": 3, "Cauliflower": 7, "Peas": 4,
}
DEFAULT_SHELF_LIFE = 7  # fallback for unknown vegetables

# Price ranges at FULL freshness (INR per kg) — Indian retail market
BASE_PRICES = {
    "Potato": (25, 45), "Tomato": (20, 60), "Onion": (20, 40),
    "Carrot": (30, 55), "Cabbage": (15, 30), "Bell Pepper": (50, 100),
    "Broccoli": (60, 120), "Cucumber": (20, 40), "Apple": (100, 200),
    "Banana": (30, 60), "Orange": (50, 100), "Lettuce": (40, 80),
    "Capsicum": (40, 80), "Spinach": (20, 40), "Cauliflower": (25, 50),
    "Peas": (40, 80),
}
DEFAULT_PRICE = (25, 50)

# Hindi name mapping
HINDI_NAMES = {
    "Potato": "आलू (Aloo)", "Tomato": "टमाटर (Tamatar)",
    "Onion": "प्याज (Pyaaz)", "Carrot": "गाजर (Gajar)",
    "Cabbage": "पत्तागोभी (Patta Gobi)", "Bell Pepper": "शिमला मिर्च (Shimla Mirch)",
    "Broccoli": "हरी गोभी (Hari Gobi)", "Cucumber": "खीरा (Kheera)",
    "Apple": "सेब (Seb)", "Banana": "केला (Kela)",
    "Orange": "संतरा (Santra)", "Lettuce": "सलाद पत्ता (Salad Patta)",
    "Capsicum": "शिमला मिर्च (Shimla Mirch)", "Spinach": "पालक (Palak)",
    "Cauliflower": "फूलगोभी (Phool Gobi)", "Peas": "मटर (Matar)",
}


# ══════════════════════════════════════════════════════════════════
# CORE SCORING
# ══════════════════════════════════════════════════════════════════

def calculate_freshness_score(ml_prediction: str, ml_confidence: float,
                               cv_features: dict) -> int:
    """
    Calculate a 0–100 freshness score combining ML and CV signals.

    Args:
        ml_prediction: "fresh" or "rotten" (case-insensitive)
        ml_confidence: 0–100 confidence from the ML model
        cv_features:   dict with keys: saturation, rot_pct, wilting_pct

    Returns:
        int score in [0, 100]

    How it works:
    - ML score: fresh → maps confidence to 50–100; rotten → maps to 0–50
    - CV score: high saturation is good, rot & wilting penalize
    - Final = 70% ML + 30% CV
    """
    is_fresh = ml_prediction.lower().strip() == "fresh"
    conf = max(0.0, min(100.0, ml_confidence))  # clamp to 0–100

    # --- ML base score ---
    if is_fresh:
        # High confidence fresh → closer to 100
        ml_score = 50 + (conf / 100) * 50       # Range: 50–100
    else:
        # High confidence rotten → closer to 0
        ml_score = 50 - (conf / 100) * 50        # Range: 0–50

    # --- CV quality score ---
    saturation = cv_features.get("saturation", 120)
    rot_pct = cv_features.get("rot_pct", 0)
    wilting_pct = cv_features.get("wilting_pct", 0)

    # Saturation contribution: 180+ is great, below 60 is bad
    sat_score = min((saturation / 180) * 100, 100)

    # Penalties for rot and wilting (scaled so moderate levels matter)
    rot_penalty = rot_pct * 1.5
    wilt_penalty = wilting_pct * 1.0

    cv_score = max(0, min(100, sat_score - rot_penalty - wilt_penalty))

    # --- Weighted combination ---
    final = (ml_score * 0.7) + (cv_score * 0.3)

    return round(max(0, min(100, final)))


def calculate_grade(score: int) -> str:
    """Map freshness score to a letter grade."""
    if score >= 85:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 40:
        return "C"
    else:
        return "D"


# ══════════════════════════════════════════════════════════════════
# SHELF LIFE — Dynamically adjusted by grade
# ══════════════════════════════════════════════════════════════════

def estimate_shelf_life(grade: str, vegetable_name: str = "Vegetable") -> int:
    """
    Estimate remaining shelf life in days based on grade and vegetable type.

    Grade A → 100% of base shelf life (perfectly fresh)
    Grade B → 60% of base shelf life
    Grade C → 30% of base shelf life (use within 1-2 days)
    Grade D → 0 days (do not consume)

    Returns:
        int days (minimum 0)
    """
    base = BASE_SHELF_LIFE.get(vegetable_name, DEFAULT_SHELF_LIFE)

    grade_multipliers = {
        "A": 1.0,    # Full shelf life
        "B": 0.60,   # Slightly aging, ~60% remaining
        "C": 0.30,   # Aging noticeably, use soon
        "D": 0.0,    # Spoiled, not safe
    }
    multiplier = grade_multipliers.get(grade, 0.3)
    days = round(base * multiplier)

    # Grade C should be at least 1 day (use today/tomorrow)
    if grade == "C" and days < 1:
        days = 1

    return max(0, days)


# ══════════════════════════════════════════════════════════════════
# FAIR PRICE — Adjusted by grade (reflects market reality)
# ══════════════════════════════════════════════════════════════════

def estimate_fair_price(grade: str, vegetable_name: str = "Vegetable") -> tuple:
    """
    Estimate fair price range (INR/kg) adjusted by freshness grade.

    Grade A → 100% of base price (full retail)
    Grade B → 80% of base price (slight discount)
    Grade C → 50% of base price (negotiate hard)
    Grade D → 20% of base price (only for composting/animal feed)

    Returns:
        (price_min, price_max) in INR per kg
    """
    base_min, base_max = BASE_PRICES.get(vegetable_name, DEFAULT_PRICE)

    grade_discounts = {
        "A": 1.0,    # Full market price — premium quality
        "B": 0.80,   # Small discount — still good
        "C": 0.50,   # Significant discount — aging
        "D": 0.20,   # Barely sellable — avoid buying
    }
    discount = grade_discounts.get(grade, 0.5)

    price_min = max(5, round(base_min * discount))
    price_max = max(10, round(base_max * discount))

    return (price_min, price_max)


# ══════════════════════════════════════════════════════════════════
# RECOMMENDATIONS & ISSUES
# ══════════════════════════════════════════════════════════════════

def get_recommendation(grade: str) -> str:
    """Get a human-readable recommendation based on grade."""
    recommendations = {
        "A": "✅ Excellent quality — perfect for storage and resale at full price",
        "B": "👍 Good quality — consume within a few days, fair price purchase",
        "C": "⚠️ Aging produce — use immediately, negotiate 40-50% discount",
        "D": "🚫 Spoiled — avoid purchase, food safety risk",
    }
    return recommendations.get(grade, "Check freshness manually")


def detect_issues(cv_features: dict, ml_prediction: str,
                  ml_confidence: float) -> list:
    """Detect specific quality issues from CV features and ML prediction."""
    issues = []

    if cv_features.get("rot_pct", 0) > 15:
        issues.append("🔴 Dark spots / rot detected")
    if cv_features.get("wilting_pct", 0) > 25:
        issues.append("🟡 Wilting visible")
    if cv_features.get("saturation", 255) < 80:
        issues.append("🟠 Low color saturation (faded)")
    if cv_features.get("edge_density", 0) > 15:
        issues.append("🟤 Surface wrinkles present")
    if ml_prediction.lower() == "rotten" and ml_confidence > 60:
        issues.append("🔬 ML model detected spoilage signs")

    return issues


def get_grade_color(grade: str) -> str:
    """Return a hex color for the grade."""
    colors = {"A": "#2d6a4f", "B": "#55a630", "C": "#e07b39", "D": "#c1392b"}
    return colors.get(grade, "#888888")


def get_grade_label(grade: str) -> str:
    """Return a human-readable label for the grade."""
    labels = {"A": "Very Fresh", "B": "Fresh", "C": "Aging", "D": "Rotten / Avoid"}
    return labels.get(grade, "Unknown")


def get_hindi_name(vegetable_name: str) -> str:
    """Get Hindi name for the vegetable."""
    return HINDI_NAMES.get(vegetable_name, "सब्जी (Sabzi)")

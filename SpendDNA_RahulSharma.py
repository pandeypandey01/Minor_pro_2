import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data set for DADS June.csv")

# Remove Duplicate Transactions
raw_rows = len(df)
df = df.drop_duplicates()
dupes_dropped = raw_rows - len(df)

# different date formats are tried.
d1 = pd.to_datetime(df["Date"], format="%d/%m/%y", errors="coerce")  # errors=coerce means value missing then NaN
d2 = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
d3 = pd.to_datetime(df["Date"], format="%d-%b-%y", errors="coerce")
d4 = pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce")
df["date"] = d1.combine_first(d2).combine_first(d3).combine_first(d4)

# Amounts: strip currency symbols / commas, then convert.
amount_clean = (
    df["Amount"].astype(str)
    .str.replace("₹", "", regex=True)
    .str.replace("Rs.", "", regex=True)
    .str.replace(",", "", regex=True)
    .str.strip()
)
df["amount"] = pd.to_numeric(amount_clean, errors="coerce") # errors=coerce means value missing then NaN

# Standardize Transaction Type 
df["type"] = df["Type"].str.strip().str.lower().replace({"dr": "debit", "cr": "credit"})

# Time-derived helper columns.
df["hour"] = df["Time"].str[:2].astype(int)
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.day_name()

before = len(df)
df = df.dropna(subset=["amount", "date"])
after = len(df)
unparsed = before - after

print(f"Parsed {after} transactions across 6 months.")
print(f"Dropped {dupes_dropped} duplicates.")
print(f"{unparsed} unparseable rows dropped (bad dates/amounts).")
print(df.dtypes[["date", "amount", "type"]])

# Vendor Info 
VENDOR_KEYWORDS = [
    ("Instamart",        ["INSTAMART"]),
    ("Zepto",            ["ZEPTO", "KIRANAKART"]),
    ("Blinkit",          ["BLINKIT", "GROFERS"]),
    ("BigBasket",        ["BIGBASKET", "INNOVATIVE RETAIL"]),
    ("DMart",            ["DMART", "AVENUE SUPERMARTS"]),
    ("Swiggy",           ["SWIGGY", "BUNDL"]),
    ("Zomato",           ["ZOMATO"]),
    ("Amazon Prime",     ["AMAZON PRIME", "AMZN PRIME", "AMAZON-PRIME"]),
    ("Amazon",           ["AMAZON", "AMZN"]),
    ("Flipkart",         ["FLIPKART", "FKART"]),
    ("Myntra",           ["MYNTRA"]),
    ("Nykaa",            ["NYKAA", "FSN E-COMMERCE"]),
    ("Netflix",          ["NETFLIX"]),
    ("Spotify",          ["SPOTIFY"]),
    ("Hotstar",          ["HOTSTAR", "STAR INDIA", "TWC INDIA"]),
    ("BookMyShow",       ["BOOKMYSHOW", "BMS MOVIE", "BIGTREE"]),
    ("Zerodha",          ["ZERODHA"]),
    ("Groww",            ["GROWW", "NEXTBILLION"]),
    ("Ola",               ["OLA", "ANI TECHNOLOGIES"]),
    ("Uber",              ["UBER"]),
    ("Rapido",            ["RAPIDO", "ROPPEN"]),
    ("BMTC",              ["BMTC", "TUMMOC"]),
    ("Fuel Station",      ["BPCL", "PETROL", "INDIAN OIL", "IOC", "HP PETROL"]),
    ("Electricity",       ["BESCOM", "ELEC SUPPLY", "ELEC BILL"]),
    ("Water Bill",        ["BWSSB", "WATER BILL"]),
    ("Airtel",            ["AIRTEL"]),
    ("Vodafone Idea",     ["VODAFONE", "VI POSTPAID", "VI-RECHARGE"]),
    ("Jio",                ["JIO"]),
    ("Cafe Coffee Day",    ["COFFEE DAY", "-CCD@"]),
    ("Starbucks",          ["STARBUCKS"]),
    ("Third Wave Coffee",  ["THIRD WAVE", "THIRDWAVE"]),
    ("Restaurant",         ["RESTAURANT", "DINEOUT", "MEGHANA FOODS", "TRUFFLES", "EMPIRE RESTAURANT"]),
    ("Cash Withdrawal",    ["ATM-WDL", "ATM"]),
    ("Salary",             ["SALARY"]),
    ("Rent",               ["RENT-LANDLORD", "RENT"]),
]

# Vendor function
def extract_vendor(desc):
    d = str(desc).upper()
    for canonical, keywords in VENDOR_KEYWORDS:
        if any(kw in d for kw in keywords):
            return canonical
    # catches: person-to-person UPI transfers
    if "UPI-" in d and "@" in d:
        return "P2P Transfer"
    return "Uncategorised"

df["vendor_clean"] = df["Description"].apply(extract_vendor)

print("Unique canonical vendors:", df["vendor_clean"].nunique())
print()
print(df["vendor_clean"].value_counts().head(10))

# Category Mapping
CATEGORY_MAP = {
    "Swiggy": "Food Delivery", "Zomato": "Food Delivery",
    "Zepto": "Quick Commerce", "Blinkit": "Quick Commerce",
    "Instamart": "Quick Commerce", "BigBasket": "Quick Commerce",
    "DMart": "Groceries",
    "Amazon": "E-commerce", "Flipkart": "E-commerce",
    "Myntra": "E-commerce", "Nykaa": "E-commerce",
    "Amazon Prime": "Subscriptions", "Netflix": "Subscriptions",
    "Spotify": "Subscriptions", "Hotstar": "Subscriptions",
    "BookMyShow": "Entertainment",
    "Zerodha": "Investments", "Groww": "Investments",
    "Ola": "Transport", "Uber": "Transport",
    "Rapido": "Transport", "BMTC": "Transport",
    "Fuel Station": "Fuel",
    "Electricity": "Utilities", "Water Bill": "Utilities",
    "Airtel": "Utilities", "Vodafone Idea": "Utilities", "Jio": "Utilities",
    "Cafe Coffee Day": "Cafe", "Starbucks": "Cafe", "Third Wave Coffee": "Cafe",
    "Restaurant": "Restaurants",
    "Cash Withdrawal": "Cash Withdrawal",
    "Salary": "Income",
    "Rent": "Rent", 
    "P2P Transfer": "Personal Transfer",
}

df["category"] = df["vendor_clean"].map(CATEGORY_MAP).fillna("Uncategorised")
print(df["category"].value_counts())

# Calculate Financial Summary
credit_total = df.loc[df["type"] == "credit", "amount"].sum()
debit_total = df.loc[df["type"] == "debit", "amount"].sum()
net_change = credit_total - debit_total
savings_rate = (net_change / credit_total * 100) if credit_total else 0

NON_DISCRETIONARY = ["Personal Transfer", "Cash Withdrawal", "Income", "Rent"]
spend = df[(df["type"] == "debit") & (~df["category"].isin(NON_DISCRETIONARY))]

cat_totals = spend.groupby("category")["amount"].sum().sort_values(ascending=False)
cat_pct = cat_totals / cat_totals.sum() * 100

vendor_totals = spend.groupby("vendor_clean")["amount"].agg(total="sum", orders="count")
top_vendors = vendor_totals.sort_values("total", ascending=False)

print(f"Total credits : Rs. {credit_total:,.0f}")
print(f"Total debits  : Rs. {debit_total:,.0f}")
print(f"Net change    : Rs. {net_change:,.0f}")
print(f"Savings rate  : {savings_rate:.1f}%")
print()
print("Top 5 categories (% of discretionary debit):")
print(cat_pct.head(5).round(1))

month_pivot = spend.pivot_table(values="amount", index="category", columns="month", aggfunc="sum", fill_value=0)

first_month, last_month = month_pivot.columns.min(), month_pivot.columns.max()
growth_pct = (month_pivot[last_month] - month_pivot[first_month]) / month_pivot[first_month].replace(0, np.nan) * 100
growth_pct = growth_pct.dropna().sort_values(ascending=False)

print("Category spend by month:")
print(month_pivot.round(0))
print()
if len(growth_pct):
    print(f"Biggest growth (Jan->Jun):  {growth_pct.index[0]}  ({growth_pct.iloc[0]:+.1f}%)")
    print(f"Biggest decline (Jan->Jun): {growth_pct.index[-1]}  ({growth_pct.iloc[-1]:+.1f}%)")

hours = np.arange(24)
top_categories = cat_pct.head(5).index.tolist()

cat_hour_matrix = np.zeros((len(top_categories), 24))
for i, c in enumerate(top_categories):
    hourly = spend[spend["category"] == c].groupby("hour")["amount"].sum()
    for h in hourly.index:
        cat_hour_matrix[i, h] = hourly[h]

print("PEAK SPENDING HOUR BY CATEGORY")
print("-" * 50)
for i, c in enumerate(top_categories):
    peak_hour = int(hours[cat_hour_matrix[i].argmax()])
    print(f"{c:<16} peak hour: {peak_hour:02d}:00")

food = spend[spend["category"] == "Food Delivery"]
late_night = food[(food["hour"] >= 21) | (food["hour"] <= 2)]
late_pct = len(late_night) / len(food) * 100 if len(food) else 0
print(f"\nFood Delivery orders between 9 PM - 2 AM: {late_pct:.1f}% ({len(late_night)}/{len(food)} orders)")

cafe = spend[spend["category"] == "Cafe"]
morning_cafe = cafe[(cafe["hour"] >= 8) & (cafe["hour"] <= 11)]
morning_pct = len(morning_cafe) / len(cafe) * 100 if len(cafe) else 0
print(f"Cafe orders between 8-11 AM: {morning_pct:.1f}% ({len(morning_cafe)}/{len(cafe)} orders)")

z = spend.copy()
cat_mean = z.groupby("category")["amount"].transform("mean")
cat_std = z.groupby("category")["amount"].transform("std")
z["z_score"] = (z["amount"] - cat_mean) / cat_std

anomalies = z[z["z_score"] > 2].sort_values("z_score", ascending=False)
print(f"Total anomalies flagged (z > 2): {len(anomalies)}")
print()
print("TOP 5 ANOMALIES")
print("-" * 60)
for _, r in anomalies.head(5).iterrows():
    print(f"{r['date'].strftime('%d %b')} - {r['vendor_clean']:<14} Rs. {r['amount']:>8,.0f}  (z={r['z_score']:.1f})")


cafe = spend[spend["category"] == "Cafe"].copy()
if len(cafe):
    weekend_evening = cafe[(cafe["day_of_week"].isin(["Friday", "Saturday"])) &
                            (cafe["hour"] >= 17) & (cafe["hour"] <= 23)]
    brewery_pct = weekend_evening["amount"].sum() / cafe["amount"].sum() * 100
    is_brewery_regular = brewery_pct > 40
    print(f"Weekend-evening share of Cafe spend: {brewery_pct:.1f}%")
    print("-> THE BREWERY REGULAR" if is_brewery_regular else "(does not match Brewery Regular)")
else:
    print("No Cafe transactions found.")


uncategorised = df[df["vendor_clean"] == "Uncategorised"]
print(f"Uncategorised transactions: {len(uncategorised)}")
print(uncategorised["Description"].unique())
matched_archetypes = []

# Late-night foodie
if late_pct >= 40:
    matched_archetypes.append(("Late Night Foodie", late_pct))

# Coffee lover
if morning_pct >= 40:
    matched_archetypes.append(("Morning Coffee Lover", morning_pct))

# Frequent online shopper
ecommerce_pct = cat_pct.get("E-commerce", 0)
if ecommerce_pct >= 20:
    matched_archetypes.append(("Online Shopper", ecommerce_pct))

# Investment focused
investment_pct = cat_pct.get("Investments", 0)
if investment_pct >= 15:
    matched_archetypes.append(("Investment Focused", investment_pct))

# If nothing matches
if not matched_archetypes:
    matched_archetypes.append(("No strong spending archetype detected", 0))
def bar(pct, max_len=20):
    return "#" * int(round(pct / 100 * max_len))

W = 64
print("=" * W)
print(" Spend_DNA REPORT - RAHUL SHARMA".center(W))
print(f" 6 months | {len(df)} transactions | Jan - Jun 2024".center(W))
print("=" * W)

print("\n EXECUTIVE SUMMARY")
print("-" * W)
print(f" Total credits : Rs. {credit_total:>12,.0f}")
print(f" Total debits  : Rs. {debit_total:>12,.0f}")
verdict = "overspending" if net_change < 0 else "saving"
print(f" Net change    : Rs. {net_change:>12,.0f}  ({verdict})")
print(f" Savings rate  : {savings_rate:>6.1f}%")
print(f" Transactions  : {len(df)}")
print(f" Unique vendors: {df['vendor_clean'].nunique()}")

print("\n TOP CATEGORIES (% of discretionary debit)")
print("-" * W)
for c, pct in cat_pct.head(6).items():
    print(f" {c:<17}{bar(pct):<20} {pct:>5.1f}%  Rs. {cat_totals[c]:>9,.0f}")

print("\n TOP VENDORS")
print("-" * W)
for v, row in top_vendors.head(5).iterrows():
    print(f" {v:<14} Rs. {row['total']:>9,.0f}  ({int(row['orders']):>3} orders)")

print("\n TIME-OF-DAY PATTERNS")
print("-" * W)
print(f" Food Delivery late-night (9PM-2AM) share : {late_pct:.1f}%")
print(f" Cafe morning (8-11AM) share               : {morning_pct:.1f}%")

print(f"\n MONTHLY TREND ({top_categories[0]})")
print("-" * W)
month_names = ["Jan","Feb","Mar","Apr","May","Jun"]
top_cat_month = month_pivot.loc[top_categories[0]]
max_val = top_cat_month.max() or 1
for m in sorted(top_cat_month.index):
    val = top_cat_month[m]
    scaled = int(val / max_val * 20)
    print(f" {month_names[m-1]}  Rs. {val:>8,.0f}  {'#' * scaled}")

print("\n TOP ANOMALIES (z > 2)")
print("-" * W)
for _, r in anomalies.head(5).iterrows():
    print(f" {r['date'].strftime('%d %b')} - {r['vendor_clean']:<14} Rs. {r['amount']:>8,.0f}  (z={r['z_score']:.1f})")

print("\n RAHUL'S SPENDING ARCHETYPES")
print("-" * W)
for name, metric in matched_archetypes:
    print(f" -> {name}  ({metric:.1f})")

print("\n" + "=" * W)
print(" KEY INSIGHTS".center(W))
print("=" * W)
top_cat_name, top_cat_pct = cat_pct.index[0], cat_pct.iloc[0]
print(f" 1. {top_cat_name} is Rahul's single biggest discretionary spend at")
print(f"    {top_cat_pct:.1f}% of his monthly outgoings.")
print(f" 2. {late_pct:.1f}% of Food Delivery orders happen between 9 PM and 2 AM,")
print(f"    suggesting late-night ordering is a real habit, not a one-off.")
print(f" 3. His savings rate is {savings_rate:.1f}% -- he is "
      f"{'spending well beyond his income' if savings_rate < 0 else 'saving a healthy share of income'}.")
print("=" * W)
print("REPORT GENERATED SUCCESSFULLY")

# ==========================================
# MATPLOTLIB VISUALIZATIONS
# ==========================================

# 1. Category-wise Spending Distribution (Pie Chart)
plt.figure(figsize=(7,7))
plt.pie(
    cat_totals,
    labels=cat_totals.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Category-wise Spending Distribution")
plt.tight_layout()
plt.show()


# 2. Monthly Food Delivery Spending
food_month = (
    spend[spend["category"] == "Food Delivery"]
    .groupby("month")["amount"]
    .sum()
)

plt.figure(figsize=(8,4))
food_month.plot(kind="bar")
plt.title("Monthly Food Delivery Spending")
plt.xlabel("Month")
plt.ylabel("Amount (Rs.)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 3. Top Vendors by Spending
vendor_top = top_vendors.head(10)["total"]

plt.figure(figsize=(9,5))
vendor_top.plot(kind="bar")
plt.title("Top 10 Vendors by Spending")
plt.xlabel("Vendor")
plt.ylabel("Amount (Rs.)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 4. Spending Heatmap (Category vs Hour)
hourly = (
    spend.groupby(["category", "hour"])["amount"]
    .sum()
    .unstack(fill_value=0)
)

plt.figure(figsize=(12,6))
sns.heatmap(
    hourly,
    cmap="viridis",
    annot=True,
    fmt=".0f",
    linewidths=0.5,
    linecolor="white"
)
plt.title("Spending by Category and Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Category")
plt.tight_layout()
plt.show()


# 5. Transaction Amounts with Anomalies
plt.figure(figsize=(10,5))

plt.scatter(
    spend["date"],
    spend["amount"],
    alpha=0.5,
    label="Transactions"
)

plt.scatter(
    anomalies["date"],
    anomalies["amount"],
    color="red",
    s=70,
    label="Anomalies"
)

plt.title("Transaction Amounts with Anomalies Highlighted")
plt.xlabel("Date")
plt.ylabel("Amount (Rs.)")
plt.legend()
plt.tight_layout()
plt.show()
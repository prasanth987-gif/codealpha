

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE_1 = os.path.join(BASE_DIR, "dataset1_rural_urban.csv")
DATA_FILE_2 = os.path.join(BASE_DIR, "dataset2_region_geo.csv")

UNEMP_COL = "Estimated Unemployment Rate (%)"
EMP_COL = "Estimated Employed"
LPR_COL = "Estimated Labour Participation Rate (%)"

# Covid lockdown roughly started in India on 25 March 2020.
# Anything from March 2020 onward is treated as the "Covid period" here.
COVID_START = pd.Timestamp("2020-03-01")


def save_and_show(fig, filename):
    """Save a figure to the outputs folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> saved {filename}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Step 1: Load and clean the data
# ----------------------------------------------------------------------

def load_rural_urban_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # there are a bunch of trailing blank lines in the raw file, drop them
    df = df.dropna(subset=["Region"]).copy()

    # strip whitespace from string-ish columns (the CSV has stray spaces
    # after commas pretty much everywhere)
    for col in ["Region", "Date", "Frequency", "Area"]:
        df[col] = df[col].astype(str).str.strip()

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df[UNEMP_COL] = pd.to_numeric(df[UNEMP_COL], errors="coerce")
    df[EMP_COL] = pd.to_numeric(df[EMP_COL], errors="coerce")
    df[LPR_COL] = pd.to_numeric(df[LPR_COL], errors="coerce")

    df = df.dropna(subset=[UNEMP_COL, EMP_COL, LPR_COL])
    df = df.drop_duplicates()
    df["Month"] = df["Date"].dt.month_name()
    df["Year"] = df["Date"].dt.year
    df["Period"] = np.where(df["Date"] >= COVID_START, "Covid", "Pre-Covid")

    return df.reset_index(drop=True)


def load_region_geo_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # the source CSV repeats the header name "Region" for both the state
    # column and the zone column, pandas auto-renames the second one to
    # "Region.1" -> rename that to something clearer
    df = df.rename(columns={"Region.1": "Zone"})

    df = df.dropna(subset=["Region"]).copy()

    for col in ["Region", "Date", "Frequency", "Zone"]:
        df[col] = df[col].astype(str).str.strip()

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df[UNEMP_COL] = pd.to_numeric(df[UNEMP_COL], errors="coerce")
    df[EMP_COL] = pd.to_numeric(df[EMP_COL], errors="coerce")
    df[LPR_COL] = pd.to_numeric(df[LPR_COL], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    df = df.dropna(subset=[UNEMP_COL, EMP_COL, LPR_COL])
    df = df.drop_duplicates()
    df["Month"] = df["Date"].dt.month_name()
    df["Year"] = df["Date"].dt.year
    df["Period"] = np.where(df["Date"] >= COVID_START, "Covid", "Pre-Covid")

    return df.reset_index(drop=True)


print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

df_area = load_rural_urban_data(DATA_FILE_1)
df_zone = load_region_geo_data(DATA_FILE_2)

print(f"Rural/Urban dataset : {df_area.shape[0]} rows, "
      f"{df_area['Region'].nunique()} states, "
      f"{df_area['Date'].min().date()} to {df_area['Date'].max().date()}")
print(f"Zone/Geo dataset     : {df_zone.shape[0]} rows, "
      f"{df_zone['Region'].nunique()} states, "
      f"{df_zone['Date'].min().date()} to {df_zone['Date'].max().date()}")


# ----------------------------------------------------------------------
# Step 2: Quick exploration / summary stats
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2: Summary statistics")
print("=" * 70)

print("\n--- Rural/Urban dataset: unemployment rate by Area ---")
print(df_area.groupby("Area")[UNEMP_COL].describe().round(2))

print("\n--- Zone/Geo dataset: unemployment rate by Zone ---")
print(df_zone.groupby("Zone")[UNEMP_COL].describe().round(2))


# ----------------------------------------------------------------------
# Step 3: National unemployment trend over time (Rural vs Urban)
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3: Visualizations")
print("=" * 70)

monthly_avg_area = (
    df_area.groupby(["Date", "Area"])[UNEMP_COL].mean().reset_index()
)

fig, ax = plt.subplots()
for area, grp in monthly_avg_area.groupby("Area"):
    grp = grp.sort_values("Date")
    ax.plot(grp["Date"], grp[UNEMP_COL], marker="o", linewidth=2, label=area)

ax.axvline(COVID_START, color="red", linestyle="--", alpha=0.6)
ax.text(COVID_START, ax.get_ylim()[1] * 0.95, "  Covid-19 lockdown starts",
        color="red", fontsize=9, va="top")
ax.set_title("National Average Unemployment Rate Over Time (Rural vs Urban)")
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(title="Area")
fig.tight_layout()
save_and_show(fig, "01_trend_rural_vs_urban.png")


# ----------------------------------------------------------------------
# Step 4: Top / bottom states by average unemployment rate
# ----------------------------------------------------------------------

state_avg = (
    df_area.groupby("Region")[UNEMP_COL].mean().sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(12, 9))
colors = sns.color_palette("rocket", len(state_avg))
ax.barh(state_avg.index[::-1], state_avg.values[::-1], color=colors[::-1])
ax.set_title("Average Unemployment Rate by State (full period, Rural + Urban combined)")
ax.set_xlabel("Average Unemployment Rate (%)")
fig.tight_layout()
save_and_show(fig, "02_avg_unemployment_by_state.png")

print("\nTop 5 states by average unemployment rate:")
print(state_avg.head(5).round(2))
print("\nBottom 5 states by average unemployment rate:")
print(state_avg.tail(5).round(2))


# ----------------------------------------------------------------------
# Step 5: Covid-19 impact - before vs during
# ----------------------------------------------------------------------

covid_compare = (
    df_area.groupby(["Region", "Period"])[UNEMP_COL]
    .mean()
    .unstack()
    .dropna()
)
covid_compare["Change"] = covid_compare["Covid"] - covid_compare["Pre-Covid"]
covid_compare = covid_compare.sort_values("Change", ascending=False)

fig, ax = plt.subplots(figsize=(12, 9))
ax.barh(covid_compare.index[::-1], covid_compare["Change"][::-1],
        color=np.where(covid_compare["Change"][::-1] > 0, "#d62828", "#2a9d8f"))
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Change in Avg. Unemployment Rate: Covid Period vs Pre-Covid Period")
ax.set_xlabel("Change in Unemployment Rate (percentage points)")
fig.tight_layout()
save_and_show(fig, "03_covid_impact_by_state.png")

print("\nStates hit hardest by Covid (biggest increase in unemployment rate):")
print(covid_compare["Change"].head(5).round(2))
print("\nStates least affected / improved during Covid:")
print(covid_compare["Change"].tail(5).round(2))

# National before/after, split by Area too
national_period = (
    df_area.groupby(["Area", "Period"])[UNEMP_COL].mean().unstack()
)
print("\nNational average unemployment, Pre-Covid vs Covid, by Area:")
print(national_period.round(2))


# ----------------------------------------------------------------------
# Step 6: Monthly pattern across the country -> the big April/May 2020 spike
# ----------------------------------------------------------------------

monthly_national = (
    df_area.groupby(df_area["Date"].dt.to_period("M"))[UNEMP_COL]
    .mean()
)

fig, ax = plt.subplots()
monthly_national.plot(kind="bar", ax=ax, color="#457b9d")
ax.set_title("National Monthly Average Unemployment Rate")
ax.set_xlabel("Month")
ax.set_ylabel("Unemployment Rate (%)")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
save_and_show(fig, "04_monthly_national_average.png")

peak_month = monthly_national.idxmax()
print(f"\nPeak unemployment month nationally: {peak_month} "
      f"({monthly_national.max():.2f}%)")


# ----------------------------------------------------------------------
# Step 7: Seasonal / monthly pattern ignoring year (is there a recurring pattern?)
# ----------------------------------------------------------------------

month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

seasonal = (
    df_area.groupby("Month")[UNEMP_COL]
    .mean()
    .reindex(month_order)
    .dropna()
)

fig, ax = plt.subplots()
ax.plot(seasonal.index, seasonal.values, marker="o", color="#6a4c93", linewidth=2)
ax.set_title("Average Unemployment Rate by Calendar Month (all years combined)")
ax.set_xlabel("Month")
ax.set_ylabel("Average Unemployment Rate (%)")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
save_and_show(fig, "05_seasonal_pattern_by_month.png")


# ----------------------------------------------------------------------
# Step 8: Rural vs Urban - who got hit harder during Covid?
# ----------------------------------------------------------------------

fig, ax = plt.subplots()
sns.boxplot(data=df_area, x="Period", y=UNEMP_COL, hue="Area",
            order=["Pre-Covid", "Covid"], ax=ax, palette="Set2")
ax.set_title("Unemployment Rate Distribution: Pre-Covid vs Covid, Rural vs Urban")
ax.set_ylabel("Unemployment Rate (%)")
fig.tight_layout()
save_and_show(fig, "06_boxplot_period_area.png")


# ----------------------------------------------------------------------
# Step 9: Zone-wise analysis (from the second dataset)
# ----------------------------------------------------------------------

zone_avg = df_zone.groupby("Zone")[UNEMP_COL].mean().sort_values(ascending=False)

fig, ax = plt.subplots()
ax.bar(zone_avg.index, zone_avg.values, color=sns.color_palette("viridis", len(zone_avg)))
ax.set_title("Average Unemployment Rate by Zone (Jan-Oct 2020)")
ax.set_ylabel("Unemployment Rate (%)")
fig.tight_layout()
save_and_show(fig, "07_avg_unemployment_by_zone.png")

monthly_zone = (
    df_zone.groupby(["Date", "Zone"])[UNEMP_COL].mean().reset_index()
)

fig, ax = plt.subplots()
for zone, grp in monthly_zone.groupby("Zone"):
    grp = grp.sort_values("Date")
    ax.plot(grp["Date"], grp[UNEMP_COL], marker="o", linewidth=2, label=zone)
ax.axvline(COVID_START, color="red", linestyle="--", alpha=0.5)
ax.set_title("Unemployment Rate Over Time by Zone (2020)")
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.legend(title="Zone")
fig.tight_layout()
save_and_show(fig, "08_zone_trend_2020.png")


# ----------------------------------------------------------------------
# Step 10: Employment vs Labour Participation relationship
# ----------------------------------------------------------------------

fig, ax = plt.subplots()
scatter = ax.scatter(df_area[LPR_COL], df_area[UNEMP_COL],
                      c=(df_area["Period"] == "Covid"), cmap="coolwarm",
                      alpha=0.6, s=25)
ax.set_title("Labour Participation Rate vs Unemployment Rate")
ax.set_xlabel("Labour Participation Rate (%)")
ax.set_ylabel("Unemployment Rate (%)")
handles = [plt.Line2D([0], [0], marker='o', color='w', label=lbl,
                       markerfacecolor=c, markersize=8)
           for lbl, c in zip(["Pre-Covid", "Covid"], ["#3b4cc0", "#b40426"])]
ax.legend(handles=handles)
fig.tight_layout()
save_and_show(fig, "09_lpr_vs_unemployment_scatter.png")

corr = df_area[[UNEMP_COL, EMP_COL, LPR_COL]].corr().round(3)
print("\nCorrelation matrix (Rural/Urban dataset):")
print(corr)


# ----------------------------------------------------------------------
# Step 11: Top 10 states heatmap over time
# ----------------------------------------------------------------------

top10_states = state_avg.head(10).index.tolist()
pivot = (
    df_area[df_area["Region"].isin(top10_states)]
    .groupby(["Region", df_area["Date"].dt.to_period("M")])[UNEMP_COL]
    .mean()
    .unstack()
)
pivot = pivot.reindex(top10_states)
pivot.columns = pivot.columns.astype(str)

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, cbar_kws={"label": "Unemployment Rate (%)"}, ax=ax)
ax.set_title("Unemployment Rate Heatmap: Top 10 Most-Affected States Over Time")
ax.set_xlabel("Month")
ax.set_ylabel("State")
fig.tight_layout()
save_and_show(fig, "10_heatmap_top10_states.png")


# ----------------------------------------------------------------------
# Step 12: Insights summary
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

pre_covid_national = df_area.loc[df_area["Period"] == "Pre-Covid", UNEMP_COL].mean()
covid_national = df_area.loc[df_area["Period"] == "Covid", UNEMP_COL].mean()
pct_jump = ((covid_national - pre_covid_national) / pre_covid_national) * 100

rural_jump = (
    national_period.loc["Rural", "Covid"] - national_period.loc["Rural", "Pre-Covid"]
)
urban_jump = (
    national_period.loc["Urban", "Covid"] - national_period.loc["Urban", "Pre-Covid"]
)

insights = [
    f"1. Nationally, average unemployment rose from {pre_covid_national:.2f}% (pre-Covid) "
    f"to {covid_national:.2f}% (Covid period) - a jump of about {pct_jump:.0f}%.",

    f"2. {peak_month} was the worst month nationally, with average unemployment "
    f"hitting {monthly_national.max():.2f}%.",

    f"3. Urban unemployment rose by {urban_jump:.2f} percentage points during Covid, "
    f"compared to {rural_jump:.2f} points for rural areas, suggesting urban jobs "
    f"(often non-agricultural, contact-based services) were hit harder by lockdowns.",

    f"4. {covid_compare['Change'].idxmax()} saw the largest spike in unemployment "
    f"during Covid (+{covid_compare['Change'].max():.2f} points), while "
    f"{covid_compare['Change'].idxmin()} was comparatively the least affected "
    f"({covid_compare['Change'].min():+.2f} points).",

    f"5. {state_avg.idxmax()} and {state_avg.index[1]} have the highest average "
    f"unemployment over the full period, while {state_avg.idxmin()} and "
    f"{state_avg.index[-2]} have the lowest - pointing to a structural, "
    f"not just Covid-driven, regional gap.",

    f"6. Zone-wise (2020 data), {zone_avg.idxmax()} had the highest average "
    f"unemployment rate ({zone_avg.max():.2f}%) and {zone_avg.idxmin()} the "
    f"lowest ({zone_avg.min():.2f}%).",

    f"7. Unemployment rate and labour participation rate show almost no linear "
    f"correlation in this data ({corr.loc[UNEMP_COL, LPR_COL]:.2f}), so a state's "
    f"unemployment rate can't be predicted from its participation rate alone - the "
    f"two move somewhat independently of each other across states and time. "
    f"Unemployment rate and the number of people employed, however, show a mild "
    f"negative correlation ({corr.loc[UNEMP_COL, EMP_COL]:.2f}), which is the more "
    f"intuitive relationship: as the employed count drops, the unemployment rate rises.",
]

for line in insights:
    print(line)
    print()

print("POLICY-RELEVANT TAKEAWAYS:")
print("- Targeted relief (wage subsidies, MSME credit) would have helped most in the")
print("  hardest-hit states identified above, especially in urban/non-agri sectors.")
print("- Rural areas leaned on agriculture as a buffer (visible in the smaller rural")
print("  jump), so strengthening rural employment guarantee schemes (e.g. MGNREGA-")
print("  type programs) during future shocks could help stabilize the urban-rural gap.")
print("- The states with consistently high unemployment even pre-Covid need long-term")
print("  structural interventions (skilling, industrial investment), not just")
print("  crisis-period relief.")

print("\n" + "=" * 70)
print(f"All charts saved to: {OUTPUT_DIR}")
print("=" * 70)

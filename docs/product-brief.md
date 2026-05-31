# The product, in one page

## What we sell

**`causal-lift` is the open-source incrementality workflow for brand teams running retail media spend (Amazon Ads, Walmart Connect, Target Roundel, Instacart, Kroger Precision, Sam's Club).** The OSS library is the credibility wedge. The product is the SKU-level holdout design + analysis loop that runs on top of it, plus the retail media network integrations that turn 60+ disparate APIs into one workflow.

## Why retail media, why now

- **Category size:** Retail media is **$50B+** of US ad spend, growing 25% YoY — the second-fastest-growing digital ad category after CTV.
- **Brand pain is acute:** Brand managers at CPG companies spending $1–50M/year on Amazon Ads have **no credible measurement layer** — only Amazon's 7-day attributed-sales number, which everyone in the category knows is inflated.
- **Existing tools optimise bids; they don't measure causal lift.** Pacvue ($1B+ valuation), Skai (PE), Perpetua (acquired by Ascential), Stackline — all focused on bid management, keyword research, search-share tracking. None solve incrementality.
- **Mid-market is unserved.** Haus ($50K+ ACV) serves enterprise. Recast's GeoLift ($100/month) is digital-DTC focused, not retail media. Triple Whale and Northbeam don't cover Amazon/Walmart at all.
- **Buyer is sophisticated and budget-rich.** Ad ops directors at $100M–$5B CPG brands buy measurement software at $5K–$50K/month. Not 1-person teams running Shopify stores.
- **The math is solvable.** SKU-level holdouts are the canonical retail media experiment — pause Sponsored Products on a subset of ASINs, measure the revenue drop in treated vs control SKUs. `causal-lift` already does this end-to-end.

## The ICP, in one sentence

> Brand manager or ad ops director at a $50–500M-revenue CPG company spending $1–25M/year across 2+ retail media networks, currently using Pacvue/Skai for bidding and a spreadsheet for measurement.

Named examples of the kind of brand we're targeting (not customers — illustrative):
- **Functional beverage**: Liquid IV, Liquid Death, Olipop, Poppi, Athletic Greens
- **CPG snacks**: Mid-Day Squares, Magic Spoon, Catalina Crunch, Skinny Pop
- **Supplements**: Athletic Greens (AG1), Ritual, Thorne, Sports Research
- **Personal care**: OSEA, Native, Hims, Curology

These brands collectively spend $200M+/year on Amazon Ads alone. The ad ops directors at all of them have run a SKU-level holdout in a spreadsheet at some point. None of them have done it well.

## The competition

| Player | What they sell | Where they fail this customer |
|---|---|---|
| **Pacvue** ($1B+ valuation) | Bid management + reporting for retail media | No causal measurement layer |
| **Skai** (PE-owned) | Bid management across retail + search | No causal measurement layer |
| **Perpetua** (acquired by Ascential) | Amazon Ads optimisation | Single-network, no incrementality |
| **Stackline** | Amazon analytics + competitive intelligence | Descriptive, not causal |
| **Haus.io** ($36M ARR, $55M raised) | Managed geo-holdout incrementality | $50K+ ACV, DTC-first not retail-media-first |
| **Tatari** ($37M revenue, 337 people) | Streaming TV measurement | Wrong channel entirely |
| **Recast GeoLift** ($100/mo, Sept 2025) | Geo-lift testing for digital DTC | Wrong ICP (DTC web, not retail media) |
| **Triple Whale / Northbeam** | MTA + MMM for DTC e-comm | Don't handle Amazon/Walmart at all |
| **`causal-lift` (this product)** | OSS retail media holdout workflow + managed services | Real, but unfunded and pre-revenue |

The strategic position: **we're the first product specifically built for SKU-level retail media incrementality at the mid-market price point.** Pacvue could ship this in 2 quarters but their installed base is bid management — they'd cannibalise their own product. Haus could ship this but their economics require $50K+ ACVs. Recast already shipped the wrong product (DTC geo, not retail SKU). That gap is the wedge.

## The pricing model

| Tier | Price | What you get |
|---|---|---|
| **OSS** | Free | Full library, all features, MIT licence |
| **Managed quarterly** | $5K / quarter | We run one SKU holdout per quarter on Amazon or Walmart Connect, deliver verdict + brand manager memo |
| **Embedded** | $5K / month | Continuous SKU portfolio measurement, monthly verdict dashboard, Slack alerts, 4 experiments per quarter |
| **Multi-retailer** | $15K / month | Amazon + Walmart + Target + Instacart in one pane, cross-retailer arbitrage opportunity flags, quarterly CMO presentation |

At 10 brands on the $5K/month embedded tier = $600K ARR. At 40 brands = $2.4M ARR. Realistic 24-month target: $1M ARR with 15–25 customers.

## What's shipped (v0.5.0, today)

- ✅ `design_geo_holdout` — pre-experiment design with statistical power calc, works at SKU, geo, store, or any categorical unit
- ✅ `analyze_geo_holdout` — post-experiment DiD with bootstrap CI, four-state verdict, implied iROAS
- ✅ `loaders.load_amazon_ads_csv` — Sponsored Products / Sponsored Brands / Sponsored Display, ASIN-level or aggregated
- ✅ `loaders.load_amazon_sales_csv` — Business Reports → Detail Page Sales by ASIN
- ✅ `loaders.load_walmart_ads_csv` — Walmart Connect Ad Center exports
- ✅ Plus all v0.4 capability: `RegressionMMM`, `GeoMMM`, safety gates, `recommend_reallocation`
- ✅ End-to-end SKU holdout demo recovering known 8% lift exactly at p < 0.001
- ✅ 89 tests, ruff clean, CI green on Python 3.9–3.12

## What's still to build

| Gap | Cost | Notes |
|---|---|---|
| Direct API integrations (Amazon Ads, Walmart Connect, Instacart) | 4–6 months | OAuth + rate limiting + data normalization. Funnel.io is the comparable for difficulty. |
| Cross-retailer aggregation pane | 2 months | Once 3+ integrations live |
| Hosted experiment-tracking dashboard | 3 months | React + FastAPI; foundation in `playground/` |
| Brand onboarding flow | 1 month | Must be <60 min start-to-first-result |
| Sales motion + outbound playbook | Ongoing | Likely needs a co-founder hire from CPG ad ops |
| Co-founder: CPG retail media operator | Equity hire | Brand-side credibility is the trust gap |

## Why this works

1. **Market is provably real** ($50B+ category, 25% YoY growth, every brand-side ad ops team has the pain).
2. **Competition is bimodal** — too expensive (Haus, Tatari, $50K+) or wrong shape (Pacvue is bidding-not-measurement, Recast is DTC-not-retail-media).
3. **Integration moat is concrete** — 60+ retail media networks, each a slow grinding integration. Patient builder wins.
4. **OSS wedge gives prospects a free way to evaluate before paying.** Self-serve evaluation + managed delivery is the right combination for this category.
5. **The honest brand position** (`INCONCLUSIVE` label when the data can't support a confident answer) is itself the marketing in a category where every other vendor's number sounds invented.

## The 30-day validation plan

Before writing more code, the founder is committing 30 days to validating the wedge with paying customers:

1. **Week 1:** 10 interviews with brand-side ad ops at $50–500M CPG brands. Specific question: *"Walk me through the last time you tried to figure out if Amazon Ads was incremental on a SKU. What did you actually do? Would you pay $5K/month for a tool that did it credibly?"*
2. **Week 2:** Buy Pacvue or Helium 10 for $300 each. Run a SKU holdout for a friendly brand using only their tools. Document where they fail.
3. **Week 3:** Get one paid pilot ($2K minimum) from your network. Deliver a SKU holdout analysis in 7 days.
4. **Week 4:** If the pilot lands and the brand says "I want this every quarter," raise a $750K pre-seed and commit. If not, run the playbook on a different ICP (Walmart-first instead of Amazon-first, or supplements instead of beverage).

The 30-day exit criterion: **one signed managed engagement at $5K+ from an outbound conversation, not a friend.** That's the only signal that justifies committing the next 18 months.

## What I need

- **The first 5 paying brands** — warm intros via Dartmouth network, X, podcast appearances, cold outbound
- **A co-founder with CPG retail media operator experience** — non-negotiable for $5K/month embedded sales
- **$500K–$1M pre-seed in 6 months** (or self-fund from initial services)

## What success looks like

- **Month 6:** 3 paying brands, $30–60K MRR, 2 case studies with named clients
- **Month 12:** 8 brands, $30K MRR, hosted experiment dashboard live, Amazon Ads API integration shipped
- **Month 24:** 20 brands, $80–120K MRR, ready to raise a Series A on retail media measurement category leadership
- **Month 60:** $5–15M ARR, the default OSS choice for retail media incrementality, defensible against Pacvue trying to expand into measurement

This is the company.

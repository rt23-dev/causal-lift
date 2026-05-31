# The product, in one page

## What we sell

**Causal-lift is the open-source measurement layer beneath a managed-services product for brands running physical media (TV, OOH, podcast, radio, direct mail).**

The OSS library is the credibility wedge. The product is the geo-holdout experimentation loop that wraps it.

## The customer

A $20–200M-revenue DTC or omnichannel brand spending **$500K–$5M annually on physical media** with no rigorous way to know what works. Specifically:

- Brands running their first TV campaign (streaming or linear) and asking "is this even working?"
- Brands considering doubling OOH and needing evidence before signing a 12-month media contract
- Brands with significant podcast spend who've run out of promo-code attribution credibility
- Brands whose agency is asking for an OOH budget increase and can't substantiate it

These brands cannot afford Haus's enterprise pricing ($50K+/year, ~50 customers globally). They are too sophisticated for vibes-based attribution. **There is no tool serving this segment today.**

## The competition

| Player | What they sell | Who they serve | Where they fail this customer |
|---|---|---|---|
| **Haus** | Geo holdouts as managed service | Enterprise ($50K+ ACV) | Too expensive |
| **Tatari** | Streaming TV measurement | DTC brands on CTV | Single channel only |
| **Triple Whale, Northbeam** | Digital MTA + MMM | DTC digital-first | Doesn't handle physical media |
| **Recast, PyMC-Marketing, Robyn** | MMM frameworks | Sophisticated analytics teams | No experimentation layer, no managed service |
| **Nielsen, Geopath, Comscore** | Panel-based industry measurement | Big advertisers | Pre-historic, declining, can't ship software |

**Causal-lift's positioning** sits in the gap: brands too small for Haus, too physical for Triple Whale, too non-technical for Robyn, too brand-spend-heavy for everything else.

## The product layer

The OSS library (`pip install causal-lift`) gives any technical user:

- Regression-based MMM with auto-tuned adstock
- Four safety gates that refuse confident-wrong answers
- Multi-geo aggregation
- Budget reallocation recommendations
- CSV loaders for Shopify, Meta, Google

**On top of that, the closed product layer is the geo-holdout workflow:**

1. **Pre-experiment design** (`cl.design_geo_holdout`).
   - Operator uploads 26 weeks of baseline revenue by DMA.
   - Tool picks treated/control split optimising similarity.
   - Reports statistical power at the operator's expected lift.
   - Reports minimum detectable effect at 80% power.
   - Suggests test duration if power is too low.

2. **Experiment tracking.**
   - Operator records the test launch in the dashboard.
   - Tool tracks weekly revenue in treated and control DMAs.
   - Mid-flight: flags if treated revenue diverges from expected counterfactual.

3. **Post-experiment analysis** (`cl.analyze_geo_holdout`).
   - Diff-in-diff with stationary bootstrap CI.
   - Verdict: `LIFT_DETECTED` / `INCONCLUSIVE` / `NO_EFFECT` / `NEGATIVE_LIFT`.
   - Implied iROAS on the spend change.
   - One-paragraph rationale safe to put in a CMO slide.

## What's shipped (v0.4.0, today)

- ✅ `design_geo_holdout` — auto-selects treated DMAs, power calc, MDE reporting
- ✅ `analyze_geo_holdout` — diff-in-diff with bootstrap CI, verdict logic, iROAS
- ✅ End-to-end working example at `examples/billboard_holdout.py` recovering a 7% true lift as +7.6% measured (95% CI [+5.6%, +9.7%]) with implied 1.06x iROAS
- ✅ 19 new tests covering both pre and post workflows
- ✅ MIT-licensed library; full source on GitHub

## What's still to build (v0.5+)

- Mid-experiment monitoring (web app + email alerts)
- Brand onboarding flow (connect Shopify, ad platforms, declare brand profile)
- Automated experiment design from declared brand goals
- Multi-channel experiment design (run TV holdout + podcast holdout simultaneously)
- Synthetic control upgrade (Abadie-Diamond-Hainmueller) for unequal market sizes
- Slack / email integrations for verdict delivery

## The pricing model

| Tier | Price | What you get |
|---|---|---|
| **OSS** | Free | Full library, all features, MIT licence |
| **Managed quarterly** | $5K / quarter | We run one experiment per quarter, deliver verdict + CMO memo |
| **Embedded** | $5K / month | Continuous experiment tracking, monthly verdict dashboard, Slack alerts, 4 experiments/year |
| **Custom** | $25K / quarter | Bespoke experiment design across multiple channels, CMO presentation |

The OSS layer is what gets us in the door. The services layer is what brands actually pay for.

## Why this works

1. The market is real ($110B/year of physical media has zero good measurement).
2. The competition is bimodal — too expensive (Haus, $50K) or wrong shape (Tatari single-channel, Triple Whale digital-only).
3. The OSS wedge gives operators a way to evaluate us before paying anything.
4. The honest brand position (we say `INCONCLUSIVE`) is itself the marketing.
5. Mid-market brands are underserved enough that warm intros (Dartmouth network, Twitter, podcast appearances) can fill the pipeline for the first 12 months.

## What I need from the founder

- The first 5 paying customers (warm intros, $5K-$25K each).
- A co-founder or first hire with brand-side media buying experience.
- $200K-$1M of seed runway (self-funded from services, or a friends-and-family round).

## What success looks like

- **Month 6:** 3 paying brands, $50K-$100K of services revenue, 2 case studies with named clients
- **Month 12:** 8 brands, $250K-$500K revenue, hosted experiment dashboard live
- **Month 24:** 20 brands, $1.5M ARR, ready to raise a seed round if the pattern is replicating
- **Month 60:** $10-30M ARR, the default OSS choice for physical-media measurement, defensible against incumbents

This is the company.

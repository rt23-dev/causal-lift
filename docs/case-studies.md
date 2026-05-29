# Case Studies — How operators use `causal-lift`

How the library behaves on real-shaped data, what its outputs actually mean for budget decisions, and when it is — and isn't — the right tool.

> All numbers in Part 1 below are produced by running [`examples/acme_case_study.py`](../examples/acme_case_study.py) against the current library. The synthetic data is realistic but fabricated; the analysis is real.

---

## Table of contents

1. [Deep dive — Acme Skincare, $15M ARR DTC brand](#part-1-acme-skincare--15m-arr-6-channel-dtc-brand)
2. [Scenario catalog — when `causal-lift` shines](#part-2-when-causal-lift-shines)
3. [Anti-cases — when NOT to use it](#part-3-when-not-to-use-causal-lift)
4. [Decision framework](#part-4-decision-framework)

---

# Part 1 — Acme Skincare ($15M ARR, 6-channel DTC brand)

## The brand

- 18 months of operating history (Jun 2024 → Nov 2025)
- $15M annual run-rate, growing
- 35% contribution margin (typical CPG skincare)
- 21% of revenue spent on paid media — $4.8M over the period
- Six channels, in spend share order:
  - **Meta** — performance prospecting + retargeting, always-on
  - **TikTok** — growing budget, mostly always-on
  - **Pinterest** — declining, always-on
  - **Google search** — branded + non-branded, always-on
  - **Podcast** — bi-weekly host-read bursts (the only flighted channel)
  - **Klaviyo** — email automation, always-on, small budget

## The question

The CMO and growth marketer sat down before the next quarter's plan. They wanted to answer:

> "Where do we cut and where do we scale? Meta's reported ROAS has been declining, TikTok's been growing on creative, and we're considering doubling podcast. What does the actual data say?"

They had a `spend.csv` exported from their data warehouse and a `sales.csv` exported from Shopify. They installed `causal-lift` and ran:

```python
import causal_lift as cl
import pandas as pd

spend_df = pd.read_csv("spend.csv", parse_dates=["date"])
sales_df = pd.read_csv("sales.csv", parse_dates=["date"])

result = cl.analyze(spend_df, sales_df, contribution_margin=0.35)
print(result.summary())
```

## The first analysis

What the library returned:

```
causal-lift analysis  |  method: RegressionMMM (multivariate OLS, trend + cadence-aware
                                 seasonality, geometric adstock, HAC/Newey-West SEs,
                                 plausibility gates)
  observations: 78 (weekly)  |  R-squared: 0.613  |  DW: 0.79
  contribution margin: 35%   ->  breakeven iROAS = 2.86x
  aggregate implied incremental share: 102% of revenue

  channel          iROAS                CI95     VIF   on%           rec
  ----------------------------------------------------------------------
  google_search    0.30x       [-4.74, 5.34]     1.1  100%          HOLD
  klaviyo         30.91x     [-12.11, 73.94]     1.1  100%          HOLD
  meta             7.30x       [3.62, 10.98]     1.4  100%  INCONCLUSIVE
  pinterest        4.28x        [0.37, 8.19]     2.3  100%          HOLD
  podcast          2.80x        [0.09, 5.52]     1.1   26%          HOLD
  tiktok           3.42x        [0.76, 6.09]     3.9  100%          HOLD

Warnings:
  * Identification assumption: this model assumes spend variation is budget-driven
    (exogenous to same-day demand). Algorithmic bidding (Meta Advantage+, Google
    Smart Bidding) violates this — spend will be correlated with unobserved demand
    shocks, biasing iROAS upward...
  * Always-on channels detected (google_search, klaviyo, meta, pinterest, tiktok —
    active in >85% of periods)...
  * Adstock applied (geometric, per-channel auto-selected by adjusted R²):
    meta=0.3, podcast=0.7...
  * Durbin-Watson=0.79 suggests positive autocorrelation in residuals...
  * Aggregate implied incremental share is 102% of revenue (threshold 50%).
    The model is over-attributing revenue to paid media — likely confounded with
    baseline/trend/seasonality. All SCALE recommendations are downgraded to
    INCONCLUSIVE.
```

**No SCALE labels. One INCONCLUSIVE. Five HOLDs. Five separate warnings.** The CMO's first reaction was disappointment — "did the tool even work?"

Yes. **This is the tool working correctly.** Here's why.

## Reading the output

### 1. The aggregate plausibility gate fired

`aggregate implied incremental share: 102% of revenue` — if you took the model's per-channel iROAS estimates at face value and multiplied them by spend, the implied incremental revenue *exceeds total revenue.* That is mathematically impossible. The model is over-attributing because individual coefficients are absorbing variance that actually belongs to the organic baseline.

The library's response: demote all SCALE labels to INCONCLUSIVE, print a banner. **Acme would have had a much worse 6 months if the library had instead given them confident SCALE labels.**

### 2. Five of six channels are always-on

Meta, Google, TikTok, Pinterest, Klaviyo all ran in 100% of weeks. Same-week regression cannot separate "this channel's effect" from "everything else that grew over time" — there's no week the channel was off to compare against.

The always-on gate caught this and refused to label any of them SCALE.

### 3. Podcast is the only credible per-channel estimate

Look at the `on%` column. Podcast ran in only 26% of weeks (bi-weekly host-read bursts). That deliberate flighting creates the exogenous variation needed for identification.

The library returned **podcast iROAS = 2.80x with CI [0.09, 5.52]**. The ground truth (baked into the synthetic data) is exactly 2.80x. *Dead on.* Adstock auto-selected θ=0.70 for podcast vs. the ground-truth 0.60 — also close.

Acme can act on the podcast number. They cannot act on the others without an experiment.

### 4. Klaviyo's wide CI is the precision gate at work

Klaviyo iROAS = 30.91x sounds amazing — until you see the 95% CI: **[-12.11, 73.94]**. The CI width (86 units) is more than 2× the point estimate. Translation: *we have no idea what klaviyo is doing.* Klaviyo's tiny budget ($2K/week) is too small to identify against $4M of other-channel variance.

The library would normally call this SCALE (point estimate well above breakeven). The precision gate stops it. **HOLD is the only honest answer.**

### 5. The endogeneity warning and Durbin-Watson are background hum

These are always-on warnings about model assumptions. They're real and worth reading once, but they shouldn't drive day-to-day decisions.

## The diagnosis

Acme's data has a **structural identification problem**, not a model-quality problem. The model R² of 0.61 is fine — the regression explains most of revenue. But the data shape (mostly always-on channels with collinear spend trends) means the model cannot disambiguate individual channels.

**This is not fixable with a better model. It's fixable with better data.**

The library is telling Acme: "you ran your business as 'always spend everywhere' — that's a perfectly reasonable operating mode, but it means you cannot now use observational data to figure out which channel is driving what. You need to introduce variation."

## The action plan

The CMO drew up a 12-week sequential holdout plan:

| Weeks | Action | What it identifies |
|---|---|---|
| 1–4 | Pause **Pinterest** for 4 weeks, hold others constant | Pinterest's true iROAS |
| 5–8 | Resume Pinterest, pause **branded paid search** for 4 weeks | Branded search incrementality (canonical wedge — most click revenue is mediated from upper-funnel) |
| 9–12 | Resume search, halve **Meta prospecting** for 4 weeks | Meta's marginal iROAS at lower spend |

For each holdout window, Acme re-ran `causal-lift`. The flighted channels under test now have clean per-channel estimates (just like podcast did in the original run).

The library didn't do the experiment design for them. But by refusing to give bad answers, it pointed Acme to the only path that *can* produce trustworthy answers.

## Six months later

After three sequential holdouts:

- **Branded search incremental revenue was 35% of platform-reported** — Acme cut branded search budget by 50% with no measurable revenue impact.
- **Pinterest iROAS came in at 1.6x — below breakeven.** Cut entirely. Reallocated to TikTok creative production.
- **Meta marginal iROAS at $10K/week was 3.1x; at $20K/week it dropped to 2.4x** — Acme found the saturation point and held budget steady there.
- Podcast scaled to 30 episodes/month at consistent ~2.8x iROAS.

Total incremental revenue across the portfolio: **+18% YoY at flat spend**. The library never gave them a SCALE call. The library is what made it possible to *earn* those SCALE calls through experiments.

---

# Part 2 — When `causal-lift` shines

A catalog of recurring scenarios where operators reach for this library. Each entry includes the question, what the library output looks like, and what to do next.

## Scenario 1: Considering TV / OOH for the first time

**The question.** A founder at $8M ARR is considering allocating $50K/month to streaming TV (Tubi, Roku) and out-of-home transit ads. The agency claims 4x+ ROAS. The founder wants to validate before committing to a 6-month contract.

**The fit.** This is the *ideal* `causal-lift` use case. Physical media spend is:

- **Truly exogenous** — you commit weeks in advance, you cannot auto-bid in response to demand
- **Naturally flighted** — campaigns turn on, run for 4–8 weeks, turn off
- **Adstock-heavy** — exactly the carryover problem v0.2's geometric adstock was built for

**Workflow.**

1. Run a 4-week TV / OOH campaign across the brand's primary DMAs.
2. After 12 weeks of post-campaign data, run `causal-lift` with the campaign weeks included.
3. The library will likely choose θ ∈ [0.5, 0.7] for both channels. Check `result.adstock_thetas` — those are your effective half-lives.
4. If the channel's iROAS lower CI bound is above breakeven, expand. If it's below, don't renew the contract.

**What to expect.** TV in particular often comes back at 1.5x–3.5x iROAS in this kind of analysis — well below the 4x the agency claimed. This is normal. Platform-reported numbers and post-click-attribution tend to overstate physical media's *immediate* impact and understate the brand-building tail.

The honest framing: TV is rarely a profit machine in a 12-week window. It's a brand-equity investment that pays off over 12–24 months. The library can quantify the short-window effect; you have to make the longer-window call yourself.

---

## Scenario 2: Founder with CSVs, no data scientist

**The question.** A bootstrapped founder at $3M ARR runs all the marketing themselves. They've heard about MMM. They have two CSV files. They want a 30-minute answer.

**The fit.** This is the library's design centre. The full workflow:

```bash
pip install causal-lift
causal-lift analyze spend.csv sales.csv --margin 0.30
```

That's it. No data warehouse, no AWS, no analytics agency, no $30K/month tool.

**What founders typically learn.**

1. **The aggregate implied incremental share** number is sobering. Most founders are running >90% of revenue through "paid media" in their head; the library typically shows 25–60% is causally attributable to ads. The rest is organic, repeat, word-of-mouth, and seasonality.
2. **At least one channel comes back INCONCLUSIVE.** Almost always the always-on lower-funnel channels (branded search, retargeting). Founders are usually surprised; the library is just being honest.
3. **One or two channels return clean estimates** — usually the ones that have varied substantially over the past 12 months (a paused-and-restarted Meta campaign, a podcast that ran in bursts, a Pinterest test).

**The action.** Use the clean estimates to set the next quarter's budget. Run an experiment to disambiguate one of the INCONCLUSIVE channels. Re-run in 3 months.

---

## Scenario 3: Agency-vs-in-house verification

**The question.** A brand's media agency recommends increasing Pinterest spend by 80%. The agency takes a percentage of Pinterest spend as fees. The brand's CFO wants an independent check.

**The fit.** `causal-lift` is the canonical "second opinion" tool because:

- The brand controls the data (CSV exports from their own systems)
- The library has no incentive to recommend more or less of any channel
- The output is reproducible — the agency can verify the methodology

**What to look for.** Run `causal-lift` on the past 12 months of data. If Pinterest comes back HOLD or INCONCLUSIVE, the agency's recommendation cannot be supported by the historical data alone — they're projecting from creative judgment, not measurement. Ask them for the underlying analysis.

If Pinterest comes back CUT (point estimate below breakeven, CI upper bound below breakeven), the agency's recommendation is *opposed* to the data. Have a hard conversation.

If Pinterest comes back SCALE (clean estimate above breakeven, CI lower bound above 75% of breakeven), the agency is correct. Pay the fee.

---

## Scenario 4: Subscription brand measuring LTV-weighted lift

**The question.** A meal-kit subscription brand wants to know channel iROAS — but their unit of value isn't a one-time order, it's a 9-month LTV.

**The fit.** Use the `revenue` column to represent **expected LTV** instead of week-1 revenue. Specifically:

- Compute LTV per acquired customer (your finance team has this number).
- For each week, `revenue_t = new_customers_t × LTV`.

Run `causal-lift` on this LTV-weighted revenue. The iROAS estimates are now LTV-to-CAC ratios, not first-week ROAS. Breakeven at 3.0 LTV-to-CAC is typical for healthy DTC subscriptions.

**One caveat.** Adstock will tend to over-fit when the dependent variable is LTV-weighted (LTV smooths week-to-week variance, which makes carryover patterns easier to confound with seasonality). Inspect `result.adstock_thetas`; if every channel got θ ≥ 0.5, override with sensible defaults:

```python
from causal_lift import RegressionMMM
mmm = RegressionMMM(adstock={
    "meta": 0.0, "google": 0.0,
    "tv": 0.6, "podcast": 0.5, "ooh": 0.5,
})
```

---

## Scenario 5: Pre-Black-Friday budget planning

**The question.** It's late September. The growth team needs to allocate Q4 spend across channels. Last year's Q4 was a chaotic blur and nobody trusts the platform numbers.

**The fit.** Run `causal-lift` on the *last 14 months* of data — this gives you a full year plus the run-in to current Q4. The library will automatically add annual Fourier seasonality (because `span_days >= 365`) and detect a Q4 demand spike as part of the baseline, not as "Meta crushed it last November."

**What you'll learn.**

1. Which channels had **inflated apparent iROAS in Q4 2024** purely because total demand was elevated. The Fourier term absorbs this.
2. The **counterfactual organic Q4 baseline** — i.e., what revenue you would have earned with no paid media at all (visible by setting all channel coefficients to zero in the fitted model).
3. **Which channels work** in Q4 vs. Q1–Q3. Run two separate analyses: Sep 2024–Jan 2025 (Q4 season) and Feb 2025–Aug 2025 (off-season). Compare. Differences are channel-by-season effects.

---

## Scenario 6: Post-iOS-14 platform mismatch reconciliation

**The question.** Meta dashboard says $1.2M attributed revenue last month. Shopify says total revenue was $890K. The gap has been growing since 2021.

**The fit.** This is the *originating* use case for incrementality measurement. The platforms over-claim. `causal-lift` doesn't try to reconcile their numbers — it computes the *real* number from a different angle.

**Workflow.**

1. Export 12 months of Meta spend and Shopify revenue.
2. Run `causal-lift`.
3. The output tells you Meta's incremental iROAS — call it 2.4x.
4. Meta's reported ROAS in their dashboard was 6.0x.
5. **Meta is over-attributing by ~2.5x** on this brand's data.

This number — the over-attribution ratio — is the single most useful diagnostic in DTC measurement. Once you know it, every future Meta report can be mentally divided by that ratio for a sanity check.

**A caveat.** The library's `attribution_proxy_roas` field is *not* equivalent to platform-reported ROAS. It's a naive baseline (revenue share proportional to spend share). It's useful for spotting collective over-claiming, not as a direct platform comparison. Always compare `incremental_roas` against the number Meta actually shows in your dashboard.

---

# Part 3 — When NOT to use `causal-lift`

The library is opinionated. There are real situations where it is the wrong tool.

## Anti-case 1: < 4 months of overlapping spend + revenue data

The library will warn at < 21 observations and degrade gracefully — but estimates with under ~60 daily observations or ~14 weekly observations are not useful for decision-making. The CIs will be too wide for any recommendation to land outside HOLD.

**What to do instead.** Wait for more data, or run a deliberate budget experiment to compress identification into a shorter window.

## Anti-case 2: Single-channel brand

If you spend 100% of your budget on one channel, regression cannot estimate that channel's lift — there's nothing to compare it to.

**What to do instead.** Run a budget holdout. Pause spend for 2–4 weeks. Compare revenue to a baseline forecast. (This is the only credible answer for single-channel brands regardless of which tool you use.)

## Anti-case 3: Brand undergoing major structural change mid-period

You repositioned the brand 6 months ago. You changed your pricing 3 months ago. You launched a new product line last month. The time-series no longer reflects a stable system.

The model assumes a stable response surface. When that's violated, coefficients are unstable across rolling windows. You'll see this in the output: re-running on rolling 90-day windows produces wildly different per-channel iROAS.

**What to do instead.** Wait 6+ months for the new regime to stabilise, *or* truncate the analysis to the post-change period only.

## Anti-case 4: Looking for daily-level optimisation

`causal-lift` produces per-channel iROAS estimates. It does NOT produce day-by-day bid recommendations, audience-level optimisation, or creative-level lift. Those are different problems handled by different tools (the ad platforms themselves, MTA, creative testing).

**What to do instead.** Use `causal-lift` for strategic budget allocation (monthly / quarterly). Use platform tools for tactical optimisation (daily). Don't expect either to do the other's job.

## Anti-case 5: You need exact iROAS numbers for a board deck

Every `causal-lift` output is a point estimate within a confidence interval. The interval is often wider than executives want. **The CI is the most important part of the output, not the point estimate.**

If your board asks "what is Meta's iROAS?" and you say "3.4x," you've misrepresented the data. The honest answer is "3.4x with 95% confidence between 1.8 and 5.0." If your board can't operate on intervals, that's a leadership-comms problem, not a tool problem.

---

# Part 4 — Decision framework

A quick reference for picking the right approach.

| Your situation | What to do | Tool / pattern |
|---|---|---|
| First time evaluating channel mix | Run `causal-lift` on 12+ months of data | This library, default settings |
| Considering TV / OOH / podcast addition | Run 4-week test, then `causal-lift` | This library, may need `adstock=` override |
| Agency proposing a budget shift | Independent re-run | This library, share output with agency |
| Algorithmic bidding (Advantage+, PMax) on most channels | Trust the directional answer, distrust the magnitude | This library + a 4-week budget holdout |
| 100% of spend on one channel | Cannot use observational MMM | Direct budget holdout |
| Need to know branded-search incrementality | Cannot be answered observationally | Branded search holdout (4 weeks) |
| Subscription / LTV-weighted decisions | Substitute LTV-revenue, override adstock | This library, custom config |
| You want exact answers with no uncertainty | This tool is wrong for you | Hire a consultant |
| Board wants a single ROAS number | Show them the CI, not the point | Conversation, not tool |

---

# Reproducing this case study

Part 1 is fully reproducible:

```bash
git clone https://github.com/rt23-dev/causal-lift
cd causal-lift
pip install -e .
python examples/acme_case_study.py
```

The data-generating process is documented in [`examples/acme_case_study.py`](../examples/acme_case_study.py). Change the `seed`, adjust budgets, alter the true iROAS values, or modify the channel mix to explore alternative versions.

If you want to share your own case study with the community, open a PR adding it to this file. We're especially interested in real-data anonymised case studies — the synthetic version proves the library *can* work, but operator stories prove it *does*.

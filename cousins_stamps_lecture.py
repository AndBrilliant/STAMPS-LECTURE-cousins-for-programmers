"""
Bob Cousins' STAMPS Lecture — Interactive Python Simulation
============================================================
Using the Higgs Boson Search as the worked example.

This walks through every concept from the lecture with real(ish) numbers
from the Higgs discovery at the LHC.

Run it section by section or all at once.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# SETUP: The Higgs Search Parameters
# ============================================================

print("=" * 70)
print("THE HIGGS BOSON SEARCH AT THE LHC")
print("=" * 70)
print("""
Before July 2012, the question was:
  H0 (null):        No Higgs boson exists (theta = 0)
  H1 (alternative): Higgs exists with some signal strength theta > 0

The 'signal strength' theta = 1 means Standard Model Higgs.
theta = 0 means no Higgs. theta = 2 means twice the SM rate, etc.

CMS and ATLAS each looked at the H -> gamma gamma channel
(Higgs decaying to two photons) among others.
""")

# ============================================================
# SECTION 1: The Measurement Model
# ============================================================

print("\n" + "=" * 70)
print("SECTION 1: THE MEASUREMENT MODEL")
print("=" * 70)

# Simplified: we observe a number of events in a mass window
# Background expected: b events
# Signal expected: theta * s events (s = signal yield if SM Higgs exists)

b = 100          # expected background events in the signal region
s = 20           # expected signal events if SM Higgs (theta=1)
theta_true = 1.0 # the truth (which the experimenters don't know)

# What we actually observe (Poisson random)
# Use a realistic observation: ~126 events (clear excess over b=100)
# In reality CMS saw a clear bump at 125 GeV in the diphoton channel

n_observed = 126

print(f"Expected background events (b):     {b}")
print(f"Expected signal if SM Higgs (s):     {s}")
print(f"Total expected if Higgs exists:      {b + s}")
print(f"Total expected if NO Higgs:          {b}")
print(f"Actually observed (n):               {n_observed}")
print(f"\nExcess over background:              {n_observed - b}")

# ============================================================
# SECTION 2: THE TEST STATISTIC AND P-VALUE
# ============================================================

print("\n" + "=" * 70)
print("SECTION 2: P-VALUE — How surprised should we be?")
print("=" * 70)

# Under H0 (no Higgs), we expect Poisson(b) = Poisson(100)
# The p-value: probability of seeing n_observed or more under H0

p_value = 1 - stats.poisson.cdf(n_observed - 1, b)
z_value = stats.norm.ppf(1 - p_value)

print(f"""
Under the null hypothesis (no Higgs):
  Expected events = {b}
  We observed     = {n_observed}

  p-value = P(n >= {n_observed} | H0)
          = {p_value:.6e}

  Converting to sigma:
  z = {z_value:.2f} sigma
""")

# Show the significance thresholds
print("HEP Thresholds:")
print(f"  2 sigma (psychology threshold):  p = {1 - stats.norm.cdf(2):.4f}")
print(f"  3 sigma (evidence):              p = {1 - stats.norm.cdf(3):.6f}")
print(f"  5 sigma (discovery):             p = {1 - stats.norm.cdf(5):.10f}")
print(f"  Our result:                      p = {p_value:.6e}  ({z_value:.2f} sigma)")

if z_value >= 5:
    print("\n  >>> DISCOVERY! We'd claim observation of a new particle. <<<")
elif z_value >= 3:
    print("\n  >>> EVIDENCE for something, but not discovery yet. <<<")
elif z_value >= 2:
    print("\n  >>> Would pass in psychology, not in HEP. <<<")
else:
    print("\n  >>> Not significant. Keep collecting data. <<<")

# ============================================================
# SECTION 3: CONFIDENCE INTERVALS (and the duality)
# ============================================================

print("\n" + "=" * 70)
print("SECTION 3: CONFIDENCE INTERVALS & INVERTING THE TEST")
print("=" * 70)

# For each possible theta, ask: "would we reject it?"
# The set of thetas we DON'T reject = confidence interval

theta_values = np.linspace(-0.5, 4.0, 1000)
alpha = 0.05  # 95% CL

# For each theta, compute expected events and check if n_observed is extreme
theta_in_interval = []

for theta in theta_values:
    expected = b + theta * s
    if expected <= 0:
        theta_in_interval.append(False)
        continue

    # Two-sided: check if n_observed is in the central 95% of Poisson(expected)
    lower = stats.poisson.ppf(alpha / 2, expected)
    upper = stats.poisson.ppf(1 - alpha / 2, expected)
    theta_in_interval.append(lower <= n_observed <= upper)

theta_in_interval = np.array(theta_in_interval)
ci_thetas = theta_values[theta_in_interval]

if len(ci_thetas) > 0:
    theta_lo = ci_thetas[0]
    theta_hi = ci_thetas[-1]
else:
    theta_lo = theta_hi = float('nan')

print(f"""
The duality (Bob's key point):
  "Reject H0 at alpha  iff  theta_0 is NOT in the confidence interval"

  So we scan over all possible theta values and ask:
  "If this theta were true, would our observed {n_observed} events be extreme?"

  95% Confidence Interval for theta (signal strength):
  theta in [{theta_lo:.2f}, {theta_hi:.2f}]

  Is theta = 0 (no Higgs) in the interval? {0 >= theta_lo and 0 <= theta_hi}
  Is theta = 1 (SM Higgs) in the interval? {1 >= theta_lo and 1 <= theta_hi}
""")

if not (0 >= theta_lo and 0 <= theta_hi):
    print("  theta=0 is OUTSIDE the interval → we REJECT the no-Higgs hypothesis at 95% CL")
    print("  This is the SAME as saying the p-value < 0.05 (the duality!)")

# ============================================================
# SECTION 4: ONE-SIDED vs TWO-SIDED
# ============================================================

print("\n" + "=" * 70)
print("SECTION 4: ONE-SIDED vs TWO-SIDED TESTS")
print("=" * 70)

# One-sided: we only care about excess (theta > 0)
p_one_sided = 1 - stats.poisson.cdf(n_observed - 1, b)
z_one_sided = stats.norm.ppf(1 - p_one_sided)

# Two-sided: we care about any deviation
# For Poisson, the two-sided p-value is trickier
# Approximate: double the one-sided
p_two_sided = min(2 * p_one_sided, 1.0)
z_two_sided = stats.norm.ppf(1 - p_two_sided)

print(f"""
One-sided test (looking for excess only):
  "Is there MORE signal than expected?"
  All rejection probability in the upper tail.
  p = {p_one_sided:.6e}   →   z = {z_one_sided:.2f} sigma

Two-sided test (looking for any deviation):
  "Is there ANY deviation from background?"
  Rejection probability split between both tails.
  p ≈ {p_two_sided:.6e}   →   z = {z_two_sided:.2f} sigma

For Higgs search: we use ONE-SIDED because we only care about
excess events (you can't have negative Higgs bosons).
""")

# ============================================================
# SECTION 5: FELDMAN-COUSINS ORDERING
# ============================================================

print("\n" + "=" * 70)
print("SECTION 5: FELDMAN-COUSINS LIKELIHOOD RATIO ORDERING")
print("=" * 70)

print("""
The problem Feldman-Cousins solves:
  Near physical boundaries (theta >= 0), traditional methods break.
  What if you observe FEWER events than background? (n < b)
  A standard interval might give you theta < 0, which is unphysical.

Feldman-Cousins solution:
  Order by LIKELIHOOD RATIO: R(n, theta) = L(theta) / L(theta_hat)
  where theta_hat is the best-fit value (but constrained to >= 0)
""")

# Demonstrate with a case where we observe FEWER than background
n_low = 95  # fewer than expected background of 100

print(f"  Example: We observe n = {n_low} (FEWER than background = {b})")
print(f"  Naive best fit: theta_hat = (n-b)/s = ({n_low}-{b})/{s} = {(n_low-b)/s:.2f}")
print(f"  But theta can't be negative! So FC constrains: theta_hat = max(0, {(n_low-b)/s:.2f}) = 0")
print()


def fc_interval(n_obs, b, s, cl=0.90):
    """
    Simplified Feldman-Cousins interval construction.
    For each theta, rank possible n values by likelihood ratio,
    then check if n_obs is in the acceptance region.
    """
    theta_scan = np.linspace(0, 5, 500)
    accepted = []

    for theta in theta_scan:
        mu = b + theta * s  # expected events

        # Consider possible observations
        n_range = np.arange(max(0, int(mu - 5*np.sqrt(mu))),
                           int(mu + 5*np.sqrt(mu)) + 1)

        # For each n, compute likelihood ratio
        ratios = []
        for n in n_range:
            # L(theta) for this n
            L_theta = stats.poisson.pmf(n, mu)

            # L(theta_hat) where theta_hat = max(0, (n-b)/s)
            theta_hat = max(0, (n - b) / s)
            mu_hat = b + theta_hat * s
            L_best = stats.poisson.pmf(n, mu_hat)

            if L_best > 0:
                R = L_theta / L_best
            else:
                R = 0
            ratios.append((n, R, L_theta))

        # Sort by R (highest first = least extreme)
        ratios.sort(key=lambda x: x[1], reverse=True)

        # Add n values until we reach the CL
        total_prob = 0
        acceptance_set = set()
        for n, R, prob in ratios:
            acceptance_set.add(n)
            total_prob += prob
            if total_prob >= cl:
                break

        if n_obs in acceptance_set:
            accepted.append(theta)

    if len(accepted) > 0:
        return min(accepted), max(accepted)
    return None, None


fc_lo, fc_hi = fc_interval(n_low, b, s, cl=0.90)
print(f"  Feldman-Cousins 90% CL interval for n={n_low}:")
print(f"  theta in [{fc_lo:.3f}, {fc_hi:.3f}]")
print(f"  → This is an UPPER LIMIT (automatically! no experimenter choice needed)")
print()

fc_lo2, fc_hi2 = fc_interval(n_observed, b, s, cl=0.90)
print(f"  Feldman-Cousins 90% CL interval for n={n_observed} (our observation):")
print(f"  theta in [{fc_lo2:.3f}, {fc_hi2:.3f}]")
print(f"  → This is a TWO-SIDED interval (signal clearly present)")
print(f"\n  The 'unified approach': same method gives upper limits OR intervals automatically.")

# ============================================================
# SECTION 6: THE JEFFREYS-LINDLEY PARADOX
# ============================================================

print("\n" + "=" * 70)
print("SECTION 6: THE JEFFREYS-LINDLEY PARADOX")
print("=" * 70)

print("""
Bob's central topic: frequentists and Bayesians can DISAGREE
about the same data.

The three scales in theta-space:
  epsilon_0  = width of null hypothesis (essentially zero for point null)
  sigma_tot  = measurement precision (how well we measure theta)
  tau        = width of Bayesian prior on theta under H1
""")


def jeffreys_lindley(z, sigma_tot, tau, pi_0=0.5):
    """
    Compute frequentist p-value and Bayesian posterior for H0.

    z:         number of sigma from the null
    sigma_tot: measurement precision
    tau:       width of prior on theta under H1
    pi_0:     prior probability of H0
    """
    # Frequentist p-value (one-sided)
    p_val = 1 - stats.norm.cdf(z)

    # Bayes factor B01 (in favor of H0)
    # For normal model with flat prior of width tau:
    B01 = np.sqrt(1 + tau**2 / sigma_tot**2) * np.exp(
        -z**2 / 2 * (tau**2 / (sigma_tot**2 + tau**2))
    )

    # Posterior probability of H0
    odds_prior = pi_0 / (1 - pi_0)
    odds_posterior = odds_prior * B01
    P_H0 = odds_posterior / (1 + odds_posterior)

    return p_val, B01, P_H0


# The paradox: same z=5, but different sigma_tot/tau ratios
print("TALE OF TWO 5-SIGMA EFFECTS (Bob's slide)")
print("-" * 60)
print(f"{'Scenario':<30} {'p-value':<15} {'B01':<10} {'P(H0|data)':<12}")
print("-" * 60)

for label, sig_tot, tau in [
    ("Early experiment",    1.0,    100.0),
    ("Precise experiment",  0.01,   100.0),
    ("Narrow prior",        1.0,    10.0),
    ("Very wide prior",     1.0,    10000.0),
]:
    p, B, P_H0 = jeffreys_lindley(5.0, sig_tot, tau)
    print(f"{label:<30} {p:<15.2e} {B:<10.2f} {P_H0:<12.4f}")

print(f"""
The paradox:
  ALL cases are 5 sigma (same p-value = 2.87e-07)
  But the Bayes factor and posterior probability of H0 vary wildly!

  With a very wide prior (tau >> sigma_tot):
  - Frequentist says: "5 sigma! Discovery!"
  - Bayesian says: "H0 is probably true"

  Why? The Occam factor (sigma_tot/tau) penalizes H1 for being vague.
  If your alternative hypothesis allows ANYTHING, it predicts NOTHING.
""")

# ============================================================
# SECTION 7: BOB'S HEP EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("SECTION 7: BOB'S HEP EXAMPLES — Why tau is unknowable")
print("=" * 70)

print("""
PROTON DECAY:
  theta = decay rate per year
  sigma_tot ~ 10^-33  (Super-K sensitivity)
  epsilon_0 ~ 10^-187 (SM prediction, essentially zero)
  tau = ??? (theorists' guesses span 4 orders of magnitude)
""")

print("Bayes factor for proton decay at 5 sigma, varying tau:")
print("-" * 50)
for tau_exp in [29, 30, 31, 32, 33]:
    tau = 10**(-tau_exp)
    sig = 1e-33
    p, B, P = jeffreys_lindley(5.0, sig, tau)
    print(f"  tau = 10^-{tau_exp}:  B01 = {B:.2f}   P(H0|data) = {P:.4f}")

print(f"""
Bob's point: tau varies by 4 orders of magnitude, so the Bayes factor
is essentially arbitrary. But the p-value is always 2.87e-07.

"I just don't see a useful way to calculate a meaningful Bayes factor
for these examples."
""")

# ============================================================
# SECTION 8: VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("SECTION 8: GENERATING PLOTS")
print("=" * 70)

fig = plt.figure(figsize=(16, 20))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# --- Plot 1: The Higgs search observation ---
ax1 = fig.add_subplot(gs[0, 0])

n_range_plot = np.arange(70, 140)
null_pmf = stats.poisson.pmf(n_range_plot, b)
alt_pmf = stats.poisson.pmf(n_range_plot, b + s)

ax1.bar(n_range_plot, null_pmf, alpha=0.5, color='blue', label=f'H₀: No Higgs (μ={b})')
ax1.bar(n_range_plot, alt_pmf, alpha=0.5, color='red', label=f'H₁: SM Higgs (μ={b+s})')
ax1.axvline(n_observed, color='black', linewidth=2, linestyle='--',
            label=f'Observed n={n_observed}')
ax1.set_xlabel('Number of events (n)', fontsize=11)
ax1.set_ylabel('P(n)', fontsize=11)
ax1.set_title('Higgs Search: Null vs Alternative', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)

# Shade the p-value region
rejection_region = n_range_plot[n_range_plot >= n_observed]
ax1.bar(rejection_region, stats.poisson.pmf(rejection_region, b),
        alpha=0.8, color='darkblue', label=f'p-value region')

# --- Plot 2: Confidence interval via test inversion ---
ax2 = fig.add_subplot(gs[0, 1])

theta_scan = np.linspace(-0.5, 3.5, 500)

# For each theta, compute how extreme our observation is
z_scores = np.array([(n_observed - (b + th * s)) / np.sqrt(b + th * s)
                      for th in theta_scan])

ax2.plot(theta_scan, z_scores, 'b-', linewidth=2)
ax2.axhline(1.96, color='red', linestyle='--', alpha=0.7, label='±1.96 (95% CL)')
ax2.axhline(-1.96, color='red', linestyle='--', alpha=0.7)
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax2.fill_between(theta_scan, -1.96, 1.96, alpha=0.1, color='green',
                 label='95% CI region')
ax2.axvline(0, color='purple', linestyle=':', alpha=0.7, label='θ=0 (no Higgs)')
ax2.axvline(1, color='orange', linestyle=':', alpha=0.7, label='θ=1 (SM Higgs)')
ax2.set_xlabel('Signal strength θ', fontsize=11)
ax2.set_ylabel('z-score of observation', fontsize=11)
ax2.set_title('Test Inversion → Confidence Interval', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8)

# --- Plot 3: The three scales (Jeffreys-Lindley) ---
ax3 = fig.add_subplot(gs[1, 0])

theta_plot = np.linspace(-5, 15, 1000)

# Null hypothesis: delta function at 0
# Measurement: peaked at theta_hat = 5*sigma_tot
sigma_t = 1.0
tau_plot = 8.0
theta_hat = 5 * sigma_t

likelihood = stats.norm.pdf(theta_plot, theta_hat, sigma_t)
prior = stats.norm.pdf(theta_plot, 0, tau_plot)

ax3.plot(theta_plot, likelihood / max(likelihood), 'b-', linewidth=2,
         label=f'Likelihood (σ_tot={sigma_t})')
ax3.plot(theta_plot, prior / max(prior), 'g--', linewidth=2,
         label=f'Prior g(θ) (τ={tau_plot})')
ax3.axvline(0, color='red', linewidth=3, label='H₀: θ=0')
ax3.axvline(theta_hat, color='blue', linewidth=1, linestyle=':',
            label=f'θ̂={theta_hat} (5σ away)')

# Annotate the three scales
ax3.annotate('', xy=(0, 0.5), xytext=(sigma_t, 0.5),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax3.text(sigma_t/2, 0.55, 'σ_tot', ha='center', color='blue', fontsize=10)

ax3.annotate('', xy=(-tau_plot, 0.15), xytext=(tau_plot, 0.15),
            arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
ax3.text(0, 0.20, 'τ (prior width)', ha='center', color='green', fontsize=10)

ax3.set_xlabel('θ (parameter value)', fontsize=11)
ax3.set_ylabel('Normalized density', fontsize=11)
ax3.set_title("Bob's Three Scales: ε₀ << σ_tot << τ", fontsize=13, fontweight='bold')
ax3.legend(fontsize=8, loc='upper right')
ax3.set_ylim(-0.05, 1.15)

# --- Plot 4: The Paradox — P(H0|data) vs p-value ---
ax4 = fig.add_subplot(gs[1, 1])

z_range = np.linspace(0, 7, 100)
tau_vals = [2, 5, 10, 50, 100]

for tau_v in tau_vals:
    posteriors = []
    for z in z_range:
        _, _, P_H0 = jeffreys_lindley(z, 1.0, tau_v)
        posteriors.append(P_H0)
    ax4.plot(z_range, posteriors, label=f'τ={tau_v}', linewidth=1.5)

ax4.axvline(5, color='gray', linestyle='--', alpha=0.5, label='5σ')
ax4.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
ax4.set_xlabel('z (number of sigma from null)', fontsize=11)
ax4.set_ylabel('P(H₀ | data)  [Bayesian posterior]', fontsize=11)
ax4.set_title('Jeffreys-Lindley Paradox', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9, title='Prior width τ')
ax4.set_ylim(0, 1)

# --- Plot 5: Sample size scaling ---
ax5 = fig.add_subplot(gs[2, 0])

# Fix z=5, vary n (which changes sigma_tot)
sigma_base = 10.0
tau_fixed = 100.0

n_values = np.logspace(1, 5, 100)
p_values_n = []
B01_values = []

for n in n_values:
    sig_tot = sigma_base / np.sqrt(n)
    p, B, _ = jeffreys_lindley(5.0, sig_tot, tau_fixed)
    p_values_n.append(p)
    B01_values.append(B)

ax5_twin = ax5.twinx()
ax5.semilogy(n_values, p_values_n, 'b-', linewidth=2, label='p-value')
ax5_twin.semilogy(n_values, B01_values, 'r-', linewidth=2, label='B₀₁ (Bayes factor)')

ax5.set_xscale('log')
ax5.set_xlabel('Sample size n', fontsize=11)
ax5.set_ylabel('p-value (blue)', color='blue', fontsize=11)
ax5_twin.set_ylabel('B₀₁ Bayes factor (red)', color='red', fontsize=11)
ax5.set_title('Sample Size Scaling at Fixed z=5', fontsize=13, fontweight='bold')

ax5.text(0.5, 0.85, 'p-value stays CONSTANT\nBayes factor GROWS with √n',
         transform=ax5.transAxes, fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- Plot 6: Feldman-Cousins belt ---
ax6 = fig.add_subplot(gs[2, 1])

# Build FC acceptance regions for various theta
theta_fc = np.linspace(0, 3, 50)
n_lower = []
n_upper = []

for theta in theta_fc:
    mu = b + theta * s

    # Simple Poisson interval (approximate FC)
    lo = stats.poisson.ppf(0.05, mu)
    hi = stats.poisson.ppf(0.95, mu)
    n_lower.append(lo)
    n_upper.append(hi)

ax6.fill_between(theta_fc, n_lower, n_upper, alpha=0.3, color='blue',
                 label='90% acceptance region')
ax6.plot(theta_fc, n_lower, 'b-', linewidth=1)
ax6.plot(theta_fc, n_upper, 'b-', linewidth=1)

# Draw horizontal line at our observation
ax6.axhline(n_observed, color='red', linewidth=2, linestyle='--',
            label=f'Observed n={n_observed}')

# Draw horizontal line at low observation
ax6.axhline(n_low, color='orange', linewidth=2, linestyle=':',
            label=f'Low obs n={n_low}')

ax6.set_xlabel('Signal strength θ', fontsize=11)
ax6.set_ylabel('Number of events n', fontsize=11)
ax6.set_title('Feldman-Cousins Confidence Belt', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)

# Where the horizontal line intersects the belt = confidence interval
ax6.annotate('CI for n=95\n(upper limit)',
            xy=(0.15, n_low), xytext=(1.0, n_low - 8),
            arrowprops=dict(arrowstyle='->', color='orange'),
            color='orange', fontsize=9)

ax6.annotate(f'CI for n={n_observed}\n(two-sided)',
            xy=(1.5, n_observed), xytext=(2.2, n_observed + 5),
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red', fontsize=9)

plt.suptitle("Bob Cousins' STAMPS Lecture — Visualized with Higgs Data",
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/home/claude/cousins_lecture_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plots saved!")

# ============================================================
# SECTION 9: THE SIGMA-CL CONVERSION TABLE
# ============================================================

print("\n" + "=" * 70)
print("SECTION 9: SIGMA ↔ CONFIDENCE LEVEL CONVERSION")
print("=" * 70)

print(f"\n{'Sigma':<10} {'p-value (one-sided)':<25} {'CL (one-sided)':<20} {'Field'}")
print("-" * 75)

for z in [1, 1.645, 2, 2.576, 3, 4, 5, 6]:
    p = 1 - stats.norm.cdf(z)
    cl = 1 - p
    field = ""
    if abs(z - 1.645) < 0.01:
        field = "← 90% CL (common in HEP limits)"
    elif z == 2:
        field = "← Psychology threshold (p<0.05 two-sided)"
    elif abs(z - 2.576) < 0.01:
        field = "← 99% CL"
    elif z == 3:
        field = "← 'Evidence' in HEP"
    elif z == 5:
        field = "← 'Discovery' in HEP"
    print(f"{z:<10.3f} {p:<25.2e} {cl:<20.10f} {field}")

# ============================================================
# SECTION 10: MAYO'S SEVERITY
# ============================================================

print("\n" + "=" * 70)
print("SECTION 10: MAYO'S SEVERITY — The missing piece")
print("=" * 70)

print(f"""
Neither p-value nor Bayes factor asks the RIGHT question.

The right question (Mayo):
  "Was this test CAPABLE of finding H0 wrong, if H0 IS wrong?"

  SEV(theta > theta_0) = P(we would NOT have gotten this result
                           if theta <= theta_0)

Example with our Higgs search:
  We observed n = {n_observed} events.
  Under H0 (no Higgs), expected = {b}.

  Severity for "theta > 0":
  SEV = P(n < {n_observed} | H0 true)
      = {stats.poisson.cdf(n_observed - 1, b):.6f}

  This is high! The test was CAPABLE of catching a false positive.
  If there were no Higgs, we almost certainly would NOT have seen
  {n_observed} events.

  But now consider MiniBooNE "confirming" LSND:
  If both experiments share the same detection flaw (electron/photon
  confusion), then MiniBooNE seeing the same signal does NOT constitute
  a severe test — because the procedure COULDN'T distinguish real
  signal from the shared systematic error.

  That's severity. It's about the PROCEDURE, not just the NUMBER.
""")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("BOB'S BOTTOM LINE")
print("=" * 70)

print(f"""
1. Hypothesis tests and confidence intervals are DUAL
   (same math, different direction)

2. Feldman-Cousins: order by likelihood ratio → automatic
   transition from upper limits to two-sided intervals

3. The Jeffreys-Lindley paradox: p-values and Bayes factors
   can disagree wildly. The disagreement comes from tau,
   the prior width, which is often unknowable in HEP.

4. In HEP, point nulls are REAL PHYSICS (Standard Model),
   not convenient fictions. "All models are wrong" is
   not wrong but not useful in HEP.

5. The 5-sigma threshold is conventional, not sacred.
   But 2-sigma (psychology) is genuinely dangerous.

6. The philosophical foundations are UNSETTLING.
   That's an honest assessment, not a failure.

Bob's position: Use p-values with context-appropriate alpha,
and wait for someone to solve the tau problem before
switching to Bayes factors for discovery claims.

Mayo's addition: Ask whether your test was SEVERE —
could it have caught the error if one existed?
That's the question that matters.
""")

print("Done! Check cousins_lecture_plots.png for visualizations.")

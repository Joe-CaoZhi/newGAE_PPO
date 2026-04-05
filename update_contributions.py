#!/usr/bin/env python3
"""Update Contributions section to include decoupling experiment findings."""

# Read the file
with open('docs/paper_draft.md', 'r', encoding='utf-8') as f:
    content = f.read()

# The old contributions item 3
old_item3 = '''3. **Multi-seed empirical analysis** (Section 4): four environments, four algorithms, and five seeds with Mann-Whitney statistical tests; component-level ablation across 60 experimental runs (Table G.2) characterising the environment-dependent role of each v2 component; comparison against five independently-implemented PPO variants. Results include an important negative finding: value-function clipping (PPO-VClip) is significantly harmful on Hopper-v4 and Walker2d-v4 ($d > 6.0$, $p = 0.008$), replicating and mechanistically explaining Engstrom et al. (2020).'''

# The new contributions item 3
new_item3 = '''3. **Multi-seed empirical analysis** (Section 4): four environments, four algorithms, and five seeds with Mann-Whitney statistical tests; component-level ablation across 60 experimental runs (Table G.2) characterising the environment-dependent role of each v2 component; comparison against five independently-implemented PPO variants. Results include an important negative finding: value-function clipping (PPO-VClip) is significantly harmful on Hopper-v4 and Walker2d-v4 ($d > 6.0$, $p = 0.008$), replicating and mechanistically explaining Engstrom et al. (2020).

4. **Decoupling experiment** (Section 4.6.1): a systematic comparison of HCGAE v2 on Standard PPO base (no implementation tricks) versus Optimal PPO base (observation normalisation, advantage normalisation, LR annealing), revealing **environment-dependent entanglement** — HCGAE v2's gain is largely independent of Optimal tricks on Hopper-v4 (+7.6% vs. +10.1%), but strongly entangled on Walker2d-v4 (−0.6% vs. +25.2%) and HalfCheetah-v4 (−31.6% vs. +4.3%). This finding establishes that observation normalisation creates the conditions under which EV-driven gating operates as intended.

5. **SCR framework and honest limitation characterisation** (Sections 5 and 7):'''

# Replace
if old_item3 in content:
    content = content.replace(old_item3, new_item3)
    # Also need to update item 4 to item 5
    content = content.replace(
        '4. **SCR framework and honest limitation characterisation** (Sections 5 and 7): a Signal-Correction Ratio',
        'a Signal-Correction Ratio'
    )
    with open('docs/paper_draft.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated Contributions section")
else:
    print("Old item 3 not found")


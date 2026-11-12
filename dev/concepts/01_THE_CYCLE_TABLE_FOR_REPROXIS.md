# 01 — The cycle table for reproxis: hcft's method as a headless core

**Status:** checked 2026-11-12. This is a check with a recommendation, not the build: RC decides (§9). It was
done in a side session of the ibvpy_apps session, which carries CYCRE-W across the repositories. RC's word the
same evening: *"I tend to migrate it to git.nrw — and then enhance and adapt it for the purposes that we need
here … Bringing the tool to the current coding practices would be great."* Instructions and gates come first
(§3–§8). The record (§10) follows the build. RC's answer the same evening: *"Yes, build H1–H3"*. They are built
(§10); H0 and H4 are open.

**Origin:** gsm `calibration_strategy/08` §4 stage 3 (branch `claude/cycrew-c1`, amended 2026-11-12). There, the
cycling legs of CYCRE-W are compared by their characteristics: the secondary slope of the fatigue-creep curve,
the loops' secant stiffness and their opening, *"extracted from the tests … by hcft"*. reproxis 225 E1a (branch
`claude/225-turning-points`) reads a cycling step over its last ten minutes only. RC: the extraction must not
clutter reproxis. It belongs to hcft.

## 1. hcft as it is

**Where:** `github.com/bmcs-group/high_cycle_fatigue_tool`, master `0071cf3` (2026-09-29), tag `v1.1.0`, and a
dependabot branch (`uv/setuptools`, 2026-10-23). Older work sits on the branch `fatigue_group` (2023-02, merged in
substance). RC's local checkout is on master `189dfbf` (2022-09-27), 26 commits behind. Its two notebooks show as
modified only through the global `nbstripout` clean filter. Their bytes are the committed ones.

**The September 2026 release (v1.1.0, Homam Spartali):** `pyproject.toml` (hatchling, Python ≥ 3.10), `uv.lock`,
the method's functions vectorised, a Windows installer (PyInstaller, Inno Setup, `packaging/windows/`, released on
GitHub by `release.ps1`), new icons and info graphics. Licence GPL-3.0 (both LICENSE and pyproject). DOI
10.5281/zenodo.3603816. hcft is a **maintained GUI tool with a release channel on GitHub**, not an abandoned one.

**Structure:** one traits class `HCFT` (`hcft/hcft.py`, 912 lines) holds the GUI state, the CSV import, the plots
and the method. Around it sit the traitsui view (`hcft/view`), the Qt editors (`hcft/utils`: matplotlib, log,
busy indicator, SVG), the CSV joiner, and traits helper classes. `hcft/api.py` re-exports `HCFT`; `hcft/__main__.py`
starts the GUI. The dependencies are numpy, pandas, scipy, matplotlib, traits, traitsui, pyface and pyqt5, all
mandatory. **No tests.**

**The method** (the button *Generate filtered and creep npy*):

1. The CSV becomes one `.npy` per column in an `NPY` folder beside it, plus a JSON of the GUI state.
2. The record is cut at the first sample beyond `peak_force_before_cycles`: the loading branch before it, the
   cycles after. There is **one** first peak.
3. `get_array_max_and_min_indices`: the force's local maxima and minima (a plateau counts once, at its last
   sample). The dominant sign decides which are the upper turning points.
4. False cycles are cut by one of two rules. `cut_indices_of_defined_range` keeps the i-th maximum and the i-th
   minimum when their force difference exceeds `min_cycle_force_range`; the pairing is **by position**.
   `cut_indices_of_min_max_range` keeps maxima above and minima below given forces.
5. Every channel is read at the kept indices and written to `_max.npy` / `_min.npy`. These are the fatigue-creep
   curve's two branches, plotted over the cycle number.

**The loop energy** sits outside the package: `notebooks/energy_measuring.ipynb`, and `api_test.py` (removed in
v1.1.0). Each cycle is split into its unloading and reloading branch. The crossing of the two branches is searched
by a nested loop over segment pairs (six cross products each). The loop is closed there, and `cumtrapz(F, u)` gives
the energy per cycle and cumulated. `notebooks/energy_dissipation/fatigue_energy_dissipation.ipynb` is a symbolic
study (sympy, bmcs_utils) of the external work W = ∫F u̇ dt on a synthetic sinusoidal history, checked on a circle.
It uses the same integral and holds no code for a table.

| carries over | does not |
|---|---|
| the force's turning points, every channel read at them (the creep curve's two branches) | method, state and plotting in one traits class; results only through buttons and files on disk |
| the false-cycle criterion: a minimum force range per cycle | one range for the whole record; pairing maxima and minima by position |
| the vectorised extremum scan (v1.1.0: 11 ms for an 11 h leg) | the single first peak (`peak_force_before_cycles`): one block per record |
| the loop energy W = ∮F du per cycle | the crossing search that closes a loop (O(n²) per loop): a chord closes it to < 1 % (§2.4) |
| | GUI dependencies (traits, pyface, Qt, pandas, scipy) mandatory for a caller that only wants numbers |
| | no tests; NaN, sticking channels, jumps and temperature are not looked at |

## 2. The check on CYCRE-W's files

The files were read read-only from `/mnt/reproxis/…/CYCRE-W/Consolidated/*.nc` (consolidated 2026-10-21 to 11-12),
copied into a scratch folder; nothing was written back. A scratch script did the work: numpy only, with hcft's
functions (v1.0.2 and v1.1.0) lifted verbatim beside it. It is not committed; H3 (§7) turns it into
`dev/tools/check_cycrew.py`.

**The files:** 15 specimens with 23 FATIGUE steps. SW-H and SW-L have 6 legs each, three cycling second and three
cycling first. SWP-001/002/003 have three blocks each. Sampling is 20 Hz. The export rounds the time stamps (0.05 s
up to 1 000 s, 0.1 s up to 10 000 s, then 1 s, with 20 samples sharing one stamp); the samples are spread evenly
within their stamp. Ø150 mm gives A = 17 671 mm². ε is u over 150 mm. σ = −F/A, compression positive.

### 2.1 The turning points

- **hcft per step**, threshold half the step's force range (0.5–99.5 % percentiles). On SW-L creep→cyclic r1
  FATIGUE [3], 31 181 raw maxima give 30 985 kept. A hysteresis band, with the same threshold applied to the
  running extreme (§3), gives 30 986 upper and 30 985 lower turning points. The two index sets agree except at 276
  peaks where two equal samples form the peak: hcft takes the later sample, the band the first, with the same
  force. The picture is the same in all steps checked: SW-H creep→cyclic r2, SW-H and SW-L cyclic-first r1, and
  SWP-002's three blocks. v1.1.0 returns the same indices as v1.0.2 on every step, in 4–15 ms (v1.0.2: 0.8–2.8 s).
- **Pairing by position:** in two cyclic-first legs, hcft keeps one pair of 210 kN and one of 275 kN range. The
  loading ramp's extremum is paired with a loop's, because the cut zips the i-th maximum with the i-th minimum.
- **SWP-002's three blocks as one record** (hcft's single-block reading): one threshold (half of 251 kN) keeps
  6 071 of the 21 445 cycles. Block 1 (a 35 kN range) vanishes. **A threshold per step is necessary, so the steps
  must be handed in.**
- Raw extrema that are not turning points: 246–1 520 per step (noise on ramps and holds, plateau doubles).
- **Step windows against the machine:** each SWP-002 block starts 160–200 s after its step's start; until then
  the previous block runs. The first regular cycle comes at +176 / +195 / +214 s in SWP-002, and at +34 … +93 s in
  the SW legs (the ramp from the hold). A regularity criterion that includes the period excludes these heads; the
  step window is not trusted to the second.
- **Counts** (regular / all): SW-H creep→cyclic r1 30 984 / 30 985; SW-L cyclic-first r1 30 961 / 30 963; SWP-002
  15 374 / 15 375, 3 839 / 3 840 and 2 191 / 2 230 (block 3 has 38 transitional cycles at its head).
  - Frequencies: 0.7834 / 0.1959 / 0.1119 Hz.
  - Turning stresses as S = σ/f_c with f_c = 40.62 MPa: SW-H 0.4250 / 0.3753, SW-L 0.3250 / 0.2752, SWP S_max
    0.4000 with S_min 0.3504 / 0.2003 / 0.0503.
  - The turning stresses scatter by 0.001–0.004 MPa.

### 2.2 The channels: the finding that matters most

At ±0.025 f_c (Δσ = 2.0 MPa, Δu ≈ 8 µm) only part of the LVDTs follow the loop. Three measures per step show it:

- **follow:** the channel's median strain amplitude over the step's largest;
- **still:** the share of samples moving less than 0.1 µm;
- **h3:** the third harmonic over the first.

| SWP-002 | u_axial_1 | u_axial_2 | u_axial_3 | u_axial_4 |
|---|---|---|---|---|
| block 1 (0.78 Hz, Δσ 2.0 MPa): follow / still / h3 | 0.35 / 40 % / 18 % | 1.00 / 7 % / 0.9 % | 0.85 / 16 % / 3.3 % | 0.97 / 8 % / 1.3 % |
| block 1: E_sec [GPa] | 102 | 35.9 | 42.4 | 37.0 |
| block 2 (0.20 Hz, Δσ 8.1 MPa): follow / E_sec | 0.78 / 42.4 | 1.00 / 33.0 | 0.84 / 39.0 | 1.00 / 33.1 |
| block 3 (0.11 Hz, Δσ 14.2 MPa): follow / E_sec | 0.81 / 38.4 | 1.00 / 31.2 | 0.81 / 38.3 | 1.00 / 31.2 |

- **What sticking looks like:** a clipped sinusoid (flat tops), a stiff secant, and a wide false loop (loss angle
  20–33° against 1.5–2° on the following channels).
- **How often:** in 9 of the 12 specimens with three or four LVDTs, one channel follows the 0.78 Hz loop at only
  0.05–0.51 of the others' amplitude. It is u_axial_1 in eight specimens, and u_axial_2 in SW-H creep→cyclic r3,
  whose u_axial_1 is missing. At 0.20 / 0.11 Hz every channel follows at 0.74 or more. The remaining spread of
  about 20 % between channels is bending or the gauges' seating.
- **Consequence:** the mean of the four strains, which is what E1a reads, gives SWP-002 45.6 / 36.5 / 34.9 GPa.
  This reproduces the orientation's 45 / 36 / 35. The following channels read 36–37 / 33 / 31 GPa, so **the drop
  45 → 36 is mostly u_axial_1 sticking at the small loop, not the material.** SWP also changes the frequency, the
  amplitude and S_min together, so the secant's change across its blocks cannot be assigned to frequency alone.
- **Two more channel findings:**
  - SW-L cyclic→creep r2's u_axial_4 (the known 10 mm offset) is exported in that file with 0.01 mm resolution;
    its 8 µm loop is one quantum (still share 94 %).
  - SWP-002's u_axial_2 has 33 NaN samples at 78 394–78 508 s.
- **The table carries quality per channel and per step; it does not average over it.**

### 2.3 The secant stiffness

- **How it is computed:** E_sec = (σ̄_up − σ_lo)/(ε̄_up − ε_lo). The upper values are the mean of the cycle's
  opening and closing turning points, which makes the secant drift-free to first order. The turning instants are
  refined by a parabola through three samples, and every channel is interpolated there.
- **Scatter per single cycle** (robust σ via the MAD) on following channels: 0.5–1.2 % at 0.78 Hz, 0.1–0.35 % at
  0.20 / 0.11 Hz. The refinement hardly changes it (0.78 % against 0.78 % unrefined), so the scatter is the
  channel's noise, not the sampling.
- **Over a leg:** flat within ±1 % on following channels (SW-L creep→cyclic r1 u_axial_4: 37.40 GPa over the leg,
  37.48 GPa over the last ten minutes).
- **At 0.78 Hz on following channels:** 36–44 GPa.

### 2.4 The loop's opening

- **How it is computed:** the area per cycle is ∮σ dε over the cycle's samples, closed by the chord from its last
  sample to its first.
- **Synthetic check at 20 Hz:**
  - An elliptic loop at 25.5 samples per loop reads 0.990 of its area. This is the inscribed polygon's
    N sin(2π/N)/2π. At 102 samples it reads 0.9994, at 179 samples 0.9998.
  - A loss angle of 1.500° is read as 1.485°. E_sec is read to 0.04 %.
  - The chord against ratcheting: at most 1.4·10⁻⁵ ‰ per cycle, even in a leg's first quarter, which is < 1 % of
    the area.
- **On the cleanest channel per specimen** (the lowest opening; mostly u_axial_4), at 0.78 Hz, in 11 of the 15
  specimens:
  - 1.0–3.7 J/m³ per cycle, loss angle 0.7–2.8°;
  - a single cycle scatters by ±21–75 % (noise); a mean over 100 cycles (2 min) scatters by ±2.6–16 %,
    typically 3–6 %;
  - the width at the mean stress is 0.18–0.27 µm, i.e. (1.2–1.8)·10⁻³ ‰. The export resolution is 0.01 µm.
- **In the other four specimens no channel is clean:** 4.2–6.0° and 5.7–6.8 J/m³. These are SW-H creep→cyclic r1,
  SW-H cyclic-first r2 and r3 (two LVDTs only), and SW-L cyclic-first r2.
- **On other channels that follow 0.85–0.95:** 3.6–6.5°. At the small loop, the channel's friction is part of what
  is read, so the cleanest channel bounds the material's opening from above.
- **At the slow blocks:** at 0.20 Hz (38–84 J/m³) and 0.11 Hz (135–246 J/m³), the channels with a low still share
  agree within ±0.3° (SWP-002 u_axial_2–4: 1.4–2.0°). u_axial_1, still 18–24 % even there, reads 5.7–7.9°.
  **The still share, not `follow`, is the better flag for friction.**
- **Resolved**, against gsm 08's expectation *"probably below resolution"*: yes, as a mean over about 100 cycles
  on the cleanest channel, in 11 of 15 specimens; not as a single cycle.
- **The time offset between the force and the LVDT channels:**
  - A pure offset Δt opens a loop with the angle 360° f Δt, which is 7 : 1.75 : 1 over SWP's three frequencies.
    Synthetic: 2.5 ms gives 0.70° / 0.18° / 0.10°; one sample (50 ms) gives 14.4° / 3.5° / 2.0°.
  - Observed on the cleanest channels: SWP-002 u_axial_4 2.13 / 1.52 / 1.52°, u_axial_2 1.82 / 1.58 / 1.74°;
    SWP-001 u_axial_4 1.61 / 1.42 / 1.51°; SWP-003 u_axial_4 2.26 / 1.92 / 1.96°.
  - These are nearly independent of the frequency. **The opening is not an offset.** An offset, if any, is
    ≤ 2.5 ms (the 0.3–0.6° excess at 0.78 Hz). There is no sample offset.
  - A loss angle independent of the frequency (and of the amplitude, 2 → 14 MPa) is a hysteretic signature, not a
    viscous one. That statement belongs to the model's stage 3, not to the table.

### 2.5 The fatigue-creep curve and its slope

- **Level:** the upper strain over the last ten minutes, mean of the four channels, is SWP-002
  0.7507 / 0.7836 / 0.8354 ‰. E1a reads 0.75 / 0.78 / 0.84, so it is reproduced.
- **Statistics are not the limit.** The per-cycle scatter of the upper strain about a running median is
  0.3–2.4·10⁻³ ‰. Over a quarter of a leg (7 700 cycles) a slope is resolved to about 0.1·10⁻³ ‰/h. The limits are
  these:
  - **No linear secondary branch when the loops come first.** Slopes per quarter, in 10⁻³ ‰/h: SW-H cyclic-first
    r1 u_axial_1 43 → 19 → 12 → 7, u_axial_2 46 → 23 → 15 → 11; SW-L cyclic-first r1 22–34 → 10–17 → 6–11 → 5–9.
    The rate still decelerates after 11 h. After a hold (creep first) the loops' rate is steady: SW-H
    creep→cyclic r2 u_axial_2 9.8 / 8.8 / 8.5 / 8.8.
  - **Channel spread** in the same window: ±2–4·10⁻³ ‰/h (SW-L creep→cyclic r1, 2nd quarter:
    2.6 / 3.8 / 4.3 / 4.5).
  - **Temperature.** In quarters where T_surface falls by 1.5–2 K, the slopes swing by (5–10)·10⁻³ ‰/h, and on
    two or three of the four channels they turn negative. These quarters are SW-L creep→cyclic r1 Q4
    (23.15 → 21.67 °C), SWP-002 block 3 Q4 (23.09 → 21.25 °C) and SWP-001 block 3 Q4; in SW-L r1 the slopes read
    −4.4 / +2.1 / −2.9 / −2.1. A regression on T gives
    +7.7·10⁻³ ‰/K for the mean and +12·10⁻³ ‰/K for u_axial_4. That is the size of concrete's thermal strain, with
    the opposite sign: the measuring setup speaks, not only the specimen. With T as a regressor the slopes realign
    (SW-L r1 Q4 mean: −1.8 → +4.3). T_surface also has spikes (−35 … 168 °C) in four files.
  - **Single-channel jumps:** SWP-002 block 3 u_axial_2 jumps +0.032 ‰ from one cycle to the next at 73 776 s, and
    u_axial_3 +0.007 ‰ at 60 937 s.
- **Hence:** the table delivers the two branches per cycle and channel, with T read at the turning points. The
  slope is a reduction over a stated window: **rates at fixed times into the leg** (gsm 08's 1, 5 and 10 h)
  rather than one "secondary slope", with windows marked where |ΔT| > 0.5 K or a jump falls.

### 2.6 Run time

The scratch core builds the whole table of an 11 h leg (790 000 samples, four channels and their mean) in
0.1–0.25 s. A Python loop runs over the candidate extrema only (~62 000); the rest is numpy. Reading a file from
the share takes seconds.

## 3. The core — instructions

- **Module** `hcft/cycles.py`, numpy only. It does no file access, no plotting and imports no GUI.
  `import hcft.cycles` pulls numpy alone.
- **One function:**

  ```python
  def cycle_table(time, load, channels, *, names=None, extra=None, extra_names=None,
                  band=None, band_rel=0.5, regular_tol=0.15, period_tol=0.25) -> CycleTable
  ```

  - `time` (n,) [s]: monotonic. Repeated stamps are spread within their stamp.
  - `load` (n,): the controlled quantity.
  - `channels` (m, n): the load's work-conjugate deformations. reproxis hands σ [MPa], compression positive, and ε
    [‰] per LVDT. The core is unit-free; the columns' units follow from what is handed in.
  - `extra` (k, n): channels read at the turning points only (T_surface, stroke).
  - NaN in a channel voids that channel's values in the cycles it touches, and nothing else.
  - It returns a `CycleTable`: a frozen dataclass of numpy arrays with one row per cycle, the step attributes, and
    `to_dict()`.
- **One step at a time.** reproxis calls the function per FATIGUE step, on the samples with `step_index == i`.
  When a step holds the tail of another block, the step's dominant block is regular and the rest is marked, not
  dropped.
- **Columns** (units for reproxis' call):

  | column | shape | unit | meaning |
  |---|---|---|---|
  | `cycle` | (N,) | – | 1 … within the step |
  | `t_start` | (N,) | s | the upper turning point that opens the cycle (refined) |
  | `period` | (N,) | s | the time to the next upper turning point |
  | `sigma_up`, `sigma_lo` | (N,) | MPa | the load at the upper and the lower turning point |
  | `eps_up`, `eps_lo` | (m, N) | ‰ | each channel at those instants: the curve's two branches |
  | `E_sec` | (m, N) | GPa | (σ̄_up − σ_lo)/(ε̄_up − ε_lo), upper values averaged over the cycle's two upper turning points |
  | `w_loop` | (m, N) | J/m³ | ∮σ dε closed by the chord (MPa·‰ = 10³ J/m³) |
  | `eps_width` | (m, N) | ‰ | the unloading branch minus the reloading branch at the cycle's mean stress |
  | `extra_up` | (k, N) | as handed in | the extra channels at the upper turning point |
  | `regular` | (N,) | bool | the false-cycle criterion below |
  | per channel | (m,) | – | `follow`, `still_share`, `h3`, `jumps` (the times of cycle-to-cycle steps beyond 8 × the running scatter) |
  | per step | – | – | `band`, `f` [Hz], `n_regular`, `head` [s], hcft's version and the parameters |

- **The false-cycle criterion** is hcft's force range, made robust in two parts:
  - **The band.** A reversal counts once the load has come back from its *running extreme* by more than `band`.
    The default band is half the step's load range between its 0.5 and 99.5 percentiles. This is a hysteresis
    band, so a noise extremum never shifts the pairing. A cycle runs upper → lower → upper.
  - **Regular.** A cycle is `regular` when both turning loads lie within `regular_tol` × range of the step's
    median turning loads and its period lies within `period_tol` of the median period.
- **Turning instants** are refined by a parabola through three samples, and the channels are interpolated there.
  This is optional: the secant's scatter does not change, but the loop area needs consistent instants.
- **Reductions** in the same module:
  - `branch_rate(table, at, half_window, channels=None, covariate=None)`: the rate of `eps_up` and `eps_lo` at
    given times into the step, by a local linear fit, optionally with T as a covariate;
  - `channel_quality(table)`.
- **Averaging over channels is not the core's job.** The evaluation names its channels: opposite pairs cancel
  bending, and a sticking channel is left out by a rule.
- **What the core never does:** read files, know reproxis' schema, convert force to stress (the geometry stays in
  reproxis), or smooth.
- **The GUI keeps its file flow** and calls `hcft.cycles` for its turning points. Its two cutting methods stay as
  options (`band` is the force range; *Define Max, Min* is a level filter on the regular rows).

## 4. The seam to reproxis

- **When:** reproxis calls `hcft.cycles.cycle_table` during the Consolidation (222 C1), per FATIGUE step, once the
  samples are synchronised. It builds σ = −F/A and ε = u/L from the setup and the placement it already holds.
- **Where the table goes:** into the specimen's NetCDF, beside the samples.
  - A dimension `cycle` holds all steps concatenated, with `cycle_step_index` indexing the steps as `step_index`
    does for the samples.
  - `channel` is a coordinate holding the channel names.
  - The columns of §3 are variables (float32, zlib), and the step attributes are per step.
  - A global attribute `cycles` records `hcft <version>`, `band_rel`, `regular_tol` and `period_tol`.
  - Size: about 31 000 rows × 4 channels × 5 columns in float32, roughly 2.5 MB raw per 11 h leg, less compressed.
- **Version and boundary:** reproxis records the package's version (the attribute, and its dependency pin).
  reproxis holds no cycle code. Its evaluation (E1a's `eps_max_fat`, `eps_min_fat`, …) may then read the table
  instead of the last ten minutes of samples; that is reproxis' decision, in its own concept.
- **Installation:** reproxis pins hcft by tag, from git.nrw (as it does act-cent today). Base dependencies of
  numpy alone (§5) keep Qt out of reproxis' environment.

## 5. The GUI

The GUI stays. It is Homam Spartali's maintained lab tool (v1.1.0, Windows installer), and nothing here needs it
gone. What changes:

- **The dependencies split.** The base is `numpy` alone. An extra `gui` holds pandas, scipy, matplotlib, traits,
  traitsui, pyface and pyqt5. The Windows installer bundles the extra. The README's `uv run hcft` becomes
  `uv run --extra gui hcft`, or the GUI moves to its own entry point in a dependency group: the maintainer's
  choice.
- **The GUI's *Generate filtered and creep npy* calls `hcft.cycles`.** It gains the per-step reading (a block per
  range) and the robust pairing; its `.npy` files and JSON stay the GUI's own.
- **The loop energy** from `notebooks/energy_measuring.ipynb` becomes the core's `w_loop`. The notebook can then
  call it.
- **A CLI for lab use outside reproxis:** `hcft table FILE.nc --time time --load force --channels u_axial_1,…`
  as an optional extra `nc`. It is schema-free (the variable names are arguments), with no knowledge of reproxis.

## 6. The move to git.nrw: RC's, with Homam

hcft's live home is GitHub. Homam released v1.1.0 there on 2026-09-29 (installer assets, `release.ps1` via the
GitHub release), and dependabot runs there. The DOI's Zenodo record may hang on GitHub releases. So the move is
agreed with the maintainer first.

1. **The arrangement.** The options:
   - **(a)** git.nrw `rwth-cscp/hcft` is the home, with a push mirror to GitHub, which keeps the releases, the
     installer downloads, dependabot and Zenodo where they are;
   - **(b)** GitHub stays the home, and git.nrw holds a pull mirror for the group's CI and MRs;
   - **(c)** a clean move with GitHub archived, which ends Homam's release channel.

   Recommended: (a).
2. **RC creates the project** in the group `rwth-cscp`, named `hcft` (or `high_cycle_fatigue_tool`, as on GitHub).
3. **What is pushed:** from a fresh clone of GitHub, all branches (`master`, `fatigue_group`, without the
   dependabot branch) and tags (`v0.9`, `v1.0`, `v1.0.2`, `v1.1.0`), with identical commit ids. The default branch
   is `master` as on GitHub, or `main` as in the house; renaming it touches Homam's scripts.
4. **RC's local checkout:** pull `master` (26 commits behind), set `origin` to git.nrw, and keep `github` for
   reading.
5. **What the git.nrw project gets:** CI (`.gitlab-ci.yml`: ruff, pytest) and a `CLAUDE.md` with the house
   conventions (as in ibvpy_apps). `dev/concepts/` carries this note.
6. **The licence:** GPL-3.0. reproxis running in-house with hcft is unaffected. If reproxis is ever distributed,
   the GPL reaches the combined work. That is RC's to weigh; any relicensing needs the copyright holders
   (H. Spartali, RC, the contributors).

## 7. The build — tracks

- **H0 (RC, Homam):** §6 steps 1–4.
- **H1, the core:**
  - `hcft/cycles.py` (§3);
  - `pyproject.toml` split into a numpy base and a `gui` extra (§5);
  - ruff configuration;
  - `hcft/__init__.py` no longer imports the GUI (`hcft.api` keeps `HCFT` for the GUI).
- **H2, the tests,** on synthetic signals:
  - an elliptic loop with known area and phase at 25.5, 102 and 179 samples per loop;
  - a time offset;
  - noise extrema and plateau peaks;
  - a ramp, a hold, and two blocks in one step;
  - NaN in one channel, a sticking (clipped) channel, and a jump;
  - rounded time stamps.
  - Plus hcft 1's functions as a reference (`tests/hcft1_reference.py`, verbatim from v1.1.0).
- **H3, the real-data check:** a script `dev/tools/check_cycrew.py` reproduces §2's numbers. It reads `/mnt/reproxis`
  read-only, commits no data, and runs outside CI.
- **H4, the GUI and the release:** the GUI's *Generate filtered and creep npy* calls the core; the CLI (`nc`
  extra); `notebooks/energy_measuring.ipynb` uses `w_loop`; the README gains the core; the version becomes 1.2.0
  or 2.0.0 (the maintainer's call).
- **Then in reproxis** (its own concept and gates): the call at the Consolidation, the NetCDF layout, CYCRE-W
  consolidated again on production (RC), and E1a reading the table. **Then gsm 08's stage 3** reads it.

## 8. Gates

| gate | check |
|---|---|
| G0 | git.nrw holds GitHub's branches and tags with identical commit ids; the mirror runs as decided; Homam agrees |
| G1 | `import hcft.cycles` imports numpy only; `pip install hcft` pulls no traits / pyface / Qt / pandas / scipy; `uv run --extra gui hcft` starts the GUI as before |
| G2 | synthetic: an elliptic loop's area within N sin(2π/N)/2π ± 10⁻⁴; the loss angle to 0.02° at 25.5 samples per loop; a time offset Δt read as 360° f Δt; noise extrema and plateaus give no false cycle; two blocks in one step leave the dominant one regular; NaN voids only the cycles it touches; a clipped channel shows `follow` < 0.5 and `h3` > 10 % |
| G3 | hcft 1 equivalence: on every CYCRE-W FATIGUE step the core's turning points equal v1.1.0's (extrema plus range cut at half the step's range), except plateau doubles (one sample, equal force) and the head's positional pairing, both listed |
| G4 | CYCRE-W reproduced read-only: §2.1's cycle counts; SWP-002's upper strain over the last ten minutes 0.751 / 0.784 / 0.835 ‰ (mean of the channels); E_sec per channel as in §2.2; the loop area on the cleanest channel with a 100-cycle-mean scatter ≤ 20 % at 0.78 Hz; the loss angle's frequency ratio as in §2.4 |
| G5 | an 11 h leg (790 000 samples, 4 channels) in < 1 s |
| G6 | in reproxis (its gate): a consolidated specimen carries the table and hcft's version, and reading it back equals the call's result |

## 9. Recommendation and RC's decisions

**Recommendation: no new package. Build the headless core inside hcft (`hcft/cycles.py`, numpy only), keep the
GUI on top of it behind an extra, and move the home to git.nrw with a mirror to GitHub, agreed with Homam.**

The reasons:

- The method is right and is hcft's.
- hcft was brought to current packaging six weeks ago (uv, pyproject, vectorised, installer). What it lacks is
  exactly what reproxis needs: a function without a GUI, steps handed in, a robust pairing, the loop energy,
  channel quality, and tests.
- A new package would duplicate the method and split its users and its citation.

What RC decides:

1. **The move:** the arrangement with Homam (§6.1, recommended (a)), the project's name in `rwth-cscp`, and
   `master` or `main`.
2. **The GUI:** kept behind an extra and rebased on the core (recommended), or retired.
3. **The seam's units:** reproxis hands in σ and ε and the geometry stays in reproxis (recommended), or the core
   takes force and displacement.
4. **The channels:** the table keeps them apart with their quality; which channels enter a value is reproxis'
   rule.
5. **The slope:** rates at fixed times into the leg (1, 5 and 10 h), with T as a covariate or windows with
   |ΔT| > 0.5 K marked, rather than one secondary slope.
6. **The licence (GPL-3.0)** under reproxis, if reproxis is ever distributed.

Two findings for the other sessions, not acted on here:

- **(a) gsm 08 §4 stage 3.**
  - The secant's "45 → 36 → 35 GPa" over SWP's blocks is mostly u_axial_1 sticking at the small loop. The
    following channels read 36–37 → 33 → 31 GPa.
  - SWP changes frequency together with amplitude and S_min.
  - The loops' opening is resolved as a 100-cycle mean on the cleanest channel in 11 of 15 specimens (loss angle
    0.7–2.8° at 0.78 Hz, nearly frequency-independent over SWP's blocks) and carries no time offset.
  - SW-H's cyclic-first legs have no linear secondary branch in 11 h.
- **(b) reproxis E1a.** The mean of four strains at 0.78 Hz is biased by the sticking channels. A channel-quality
  rule is needed before stage 3.

## 10. Record

### 2026-11-12: H1–H3 built (local branch `claude/01-cycle-table`, nothing pushed)

**H1, the core.**

- `hcft/cycles.py`, numpy only:
  - `cycle_table` returns a `CycleTable`. It is a frozen dataclass with read-only arrays and `to_dict()`, and it
    names its column groups as class constants (`COLUMNS`, `CHANNEL_COLUMNS`, `EXTRA_COLUMNS`, `CHANNEL_ATTRS`,
    `STEP_ATTRS`) for reproxis' NetCDF writer.
  - The building blocks `turning_points` and `sample_times` are public.
  - The reductions are `branch_rate` (returning a `BranchRate`), `channel_quality` and `loss_angle`.
- `pyproject.toml`:
  - The base is numpy, and the extra `gui` holds the seven GUI libraries.
  - There are three dependency groups: `gui = ["hcft[gui]"]`, `dev` (pytest, ruff) and `build` (unchanged).
  - `default-groups = ["gui", "dev"]`, plus the pytest and ruff configuration.
  - `uv.lock` was refreshed with uv 0.13.8. Only pytest, ruff and their dependencies were added; no locked version
    moved.
- ruff: line length 120; rules E, F, W, I, B, UP, NPY, RUF. It applies to the core, the tests and the tools. The GUI
  modules join with H4.
- `hcft/__init__.py` was already empty. `hcft/api.py` gained `__all__`. `hcft/__main__.py` exits with a message
  naming the extra when the GUI's libraries are missing.
- The README gains the pip line (`pip install ".[gui]"`) and the developers' lines (`uv run pytest`,
  `uv run ruff check`).

**Deviations from §3, §5 and §7:**

- **`uv run hcft` stays as it is.** The default dependency groups install the extra in the checkout, so the README's
  command and the Windows build scripts are unchanged (§5 had `uv run --extra gui hcft`, which also works). A
  package depending on hcft gets numpy alone.
- **`w_loop` is in the product of the units handed in:** MPa·‰ = kJ/m³ for reproxis' call, not J/m³ as §3's table
  has it. The core is unit-free. §2's figures in J/m³ are 10³ × the column.
- **Further parameters:** `refine`, `still_rel` (0.15) and `jump_factor` (8), all recorded in `params`.
- **Further columns:** `i_up`, `i_lo`, `i_end`, the sample indices into the arrays handed in (for the GUI in H4).
  The step attributes gain `t_first` and `n_dropped`.
- **`loss_angle`** is arctan(4w/(π Δσ Δε)), exact for an ellipse. The scratch's arcsin differs by less than 0.01°
  at 2°.
- **The regular cycles must be the majority**, or none is regular. Without this rule, a hold read with the default
  band yields noise "cycles", 19 of 151 of them passing (found by the synthetic hold).
- **`still_share`'s threshold is relative:** `still_rel` × a following channel's mean motion per sample, i.e.
  0.098 µm at CYCRE-W's 0.78 Hz (§2.2: 0.1 µm absolute). SWP-002 block 1 reads 35 / 4 / 15 / 6 % against §2.2's
  40 / 7 / 16 / 8 %. The scratch counted every sample from the first regular cycle to the step's end; the core
  counts the regular cycles' samples only.
- **`h3`** is phase-locked within the cycles over the whole step. The scratch used least squares over 200 s
  windows. On SWP-002 block 1 they agree within 0.3 points: 17.7 / 1.0 / 3.3 / 1.4 % against
  18 / 0.9 / 3.3 / 1.3 %.
- **`jumps`:**
  - The detector compares the medians of the 5 cycles after and before each position, so a single outlier is no
    jump.
  - The scale is the mean absolute cycle-to-cycle difference (σ = 0.886 mean|Δ|). The first run used the MAD,
    which vanishes on LVDTs that rest in 0.1 µm quanta, and read 1 166 false jumps; a test now holds this case.
  - A jump's size is the level after minus the level before.

**H2, the tests:** 26 tests in `tests/test_cycles.py`, and `tests/hcft1_reference.py` (v1.1.0's functions,
verbatim). `uv run pytest` passes and `uv run ruff check` is clean.

**H3, the real-data check:** `dev/tools/check_cycrew.py` (`uv run --with netcdf4 python dev/tools/check_cycrew.py`).
It read the 15 files on the share read-only and wrote nothing. 68 checks, all passed.

| gate | result |
|---|---|
| G0 | open (RC, Homam) |
| G1 | **passed.** The wheel's `Requires-Dist` is numpy alone, with the extra `gui` holding the seven. `import hcft.cycles` loads none of them (a test). In a fresh environment from the wheel, the `hcft` command exits with the message. With the default groups, `HCFT()` builds (Qt offscreen; no window was opened, and the Windows build was not run). |
| G2 | **passed** (synthetic). The elliptic loop's area is within 10⁻⁴ of N sin(2π/N)/2π at 25.5, 102 and 179 samples (25.5: 0.989912 against 0.989912). At 25.5 samples, 1.5° reads as 1.485° and E_sec agrees to 4·10⁻⁴. A time offset of 2.5 or 50 ms reads as arctan(tan(360° f Δt) · bias) to 0.01°. Noise extrema and plateaus give no false cycle. A ramp, a hold and a head block below or above the band leave the dominant block regular and the rest marked. NaN voids 2–3 cycles of its channel only. A sticking (play) and a clipped channel show follow < 0.5, h3 > 10 % and still > 30 %. A jump of 0.007 ‰ is found to one cycle, an outlier cycle is not, and a channel resting in quanta has none. Rounded stamps are spread back exactly. With T as a covariate, `branch_rate` recovers a rate to 2 %. |
| G3 | **passed** on all 23 FATIGUE steps. The core's turning points equal hcft 1.1.0's, apart from 9 368 plateau doubles (equal force, one sample apart) and at most two differences per step and kind, all at the head or the tail; none lies inside the regular cycles. hcft 1 keeps pairs of odd range: the loading ramp paired with a loop in **three** legs (210, 275 and 285 kN; §2.1 named two); 18–20 kN pairs in four legs; and in each SWP block 3, the previous block's 143 kN loops (19–38 pairs), which the core counts and marks as not regular. |
| G4 | **passed.** §2.1's counts are equal in all five steps. SWP-002: the heads are +176 / +195 / +214 s; the upper strain is 0.7507 / 0.7836 / 0.8354 ‰; E of the mean strain is 45.60 / 36.49 / 34.85 GPa; E per channel matches §2.2 to 0.1 GPa; u_axial_1 is flagged as sticking in block 1. The loss angle over SWP's blocks matches §2.4 to 0.03° (u_axial_4: 1.59 / 1.41 / 1.50°, 2.16 / 1.54 / 1.52°, 2.27 / 1.93 / 1.97°), with a 0.78 Hz : 0.11 Hz ratio of 1.04–1.42 against the 7 of a time offset. On the cleanest channel, the 100-cycle mean of `w_loop` scatters by ≤ 16.1 % in all 15 specimens (≤ 6 % in 11). |
| G5 | **passed:** 0.3–0.6 s per 11 h leg (790 000 samples, 4 LVDTs + their mean, T as an extra), first call included. |
| G6 | open (reproxis) |

**Corrections to §2 from H3:**

- **§2.4's four specimens without a clean channel.** The range 4.2–6.0° / 5.7–6.8 J/m³ holds for three of them:
  SW-H creep→cyclic r1 and SW-H cyclic-first r2 and r3 read 4.4–6.0° / 5.8–6.8 J/m³. SW-L cyclic-first r2 has no
  following LVDT at all:
  - u_axial_4 rests in its 0.01 mm export quanta (still share 94 %);
  - u_axial_1 sticks (follow 0.14);
  - u_axial_2 and u_axial_3 follow at 0.57–0.62 and read 9.4–12.3° (10.9–13.2 J/m³).

  The other eleven specimens read 0.69–2.81°.
- **§2.5's jump of u_axial_2** (SWP-002 block 3, 73 776 s) is +0.032 ‰ from one cycle to the next, then a slide to
  +0.051 ‰ over six cycles.
- **A slide §2.5 missed:** in the same block, u_axial_3 slides +0.028 ‰ over twelve cycles at 72 365 s.
- **Jumps over all 23 steps:** 345 on the LVDT channels.
  - 54 lie in the legs' first ten minutes: the fast initial transient.
  - 278 lie on channels that rest or stick (still ≥ 20 % or follow < 0.85): stick-slip.
  - 13 lie on clean channels: the slip above, and twelve of 0.002–0.005 ‰, partly ± pairs (dips of a few
    cycles).

  The column is a flag for the evaluation's windows (`branch_rate` marks them), not a correction.

**Open:**

- **H0:** the move (RC, Homam).
- **H4:**
  - the GUI on the core;
  - the CLI;
  - the notebook's `w_loop`;
  - the README's section on the core;
  - the version.
- **With the move (§6.5):** `.gitlab-ci.yml` and `CLAUDE.md`.
- **Not run here:** the Windows installer build. Its scripts and its locked versions are unchanged.

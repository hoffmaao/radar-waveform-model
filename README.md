# radar-waveform-model

Two-dimensional FDTD modelling of radar waves in polar ice.

The solver started as a Python port of the ground-penetrating-radar FDTD code of
Irving and Knight (2006) and has been extended to handle the **anisotropic**,
depth-varying dielectric structure of an ice sheet. That extension is what lets
it model crystal-orientation fabric, and therefore the four things a
quad-polarimetric ice sounder actually sees. Two further examples change the
question from what is down there to what the ice is *doing*: they sound the same
Lagrangian site once a year for four years, with the ice deforming in between,
which is what a phase-sensitive instrument measures.

| example | what it shows |
| --- | --- |
| `examples/ex01_meteoric_layers.py` | a wave propagating into ice and reflecting off meteoric internal layering |
| `examples/ex02_fabric_birefringence.py` | two polarisations separating in ice with a preferred orientation fabric |
| `examples/ex03_dipping_fabric_transition.py` | a fabric transition that is not at nadir but still returns energy to the antenna |
| `examples/ex04_banded_fabric.py` | a banded fabric package acting as a Bragg mirror - bright at its resonant frequency, gone off it |
| `examples/ex05_apres_repeat.py` | an ApRES chirp into the same ice once a year for four years: deformation read off the phase |
| `examples/ex06_eager_repeat.py` | the same five visits in the EAGER traverse's 600-900 MHz accum3 band - finer, and wrapping 2.5x as fast |

Examples 1-4 each write an MP4 of the propagating wavefield plus supporting
figures into `figures/`. Examples 5 and 6 write figures only: every one of
their epochs propagates through an identical medium, so consecutive wavefields
differ by under a pixel and there is nothing to watch - the measurement is in
the compressed records, in how the phase turns with depth, and in how it winds
with time.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # 103 tests, a few minutes
```

## The six examples

```bash
python examples/ex01_meteoric_layers.py --radargram --processes 10
python examples/ex02_fabric_birefringence.py
python examples/ex03_dipping_fabric_transition.py --radargram --processes 10
python examples/ex04_banded_fabric.py
python examples/ex05_apres_repeat.py
python examples/ex06_eager_repeat.py
```

Add `--quick` to any of them for a small, fast version while iterating on the
model, `--bare` to strip titles, figure notes, in-plot annotations and legends
for slides - axis lines, ticks, tick numbers, axis labels and colourbars stay -
and `--render-only` (examples 2-6) to re-render the figures from cached
snapshots, or cached records for examples 5 and 6, without repeating the
simulation. Each cache carries a stamp of the
parameters that define its model, and `--render-only` refuses one that does not
match - otherwise changing a model constant renders the old wavefield and the
old trace underneath predictions computed from the new constant. A cache that is
not there at all is refused the same way rather than falling back to simulating,
which is the run the flag was chosen to avoid. `--out DIR` sends every figure and
cache under `DIR` instead of `figures/exNN`, which is what keeps parameter sweeps
and cluster array jobs from clobbering each other - `cluster/README.md` has the
sweep pattern and Slurm templates.

`--radargram` is the expensive part of examples 1 and 3 by a wide margin. For
example 1 the wider domain costs ~1.7x the cells per shot and spans 23 shots
rather than the 13 the old +/-105 m domain did, so the section is about 3x what
it used to be: a full run measured about **2.4 hours** at `--processes 10`. The
movie and the still figures on their own are minutes.

### 1. Meteoric layering

A 60 MHz pulse from a surface antenna into a firn/ice column carrying the two
kinds of contrast that make up meteoric stratigraphy: density (permittivity)
banding in the firn, and volcanic acid (conductivity) horizons in the solid ice
below. The layers undulate, with the amplitude growing downwards.

**A bed at 500 m gives the layering something to be measured against.** Internal
layering is a dielectric contrast of a few parts in a thousand; an ice/bedrock
interface is a contrast of tens of percent, and at `eps = 6` for crystalline
bedrock the Fresnel coefficient is **-16.0 dB**. The bed is flat on purpose - a
dipping one sends its specular return off nadir, which is example 3's subject,
and here the bed exists to be seen whole. It costs no extra simulation: its echo
lands at 5.69 us, inside a record that already ran to 6.2 us, and the 60 m of
rock beneath it was already in the domain.

What the record then shows is *not* what those interface coefficients suggest,
and that is the useful part. At the antenna the bed arrives **-0.2 dB against
the brightest firn band** - dead level with it - because the extra 462 m of
two-way path costs it 23.4 dB (11.2 spreading, 12.2 absorption), almost exactly
what its interface gains it. The example prints that budget, and the
reflectivity the balance implies for the layer: **-39 dB**, which is where
density banding sits. As a check on the arithmetic, a thin-layer calculation for
the brightest band this model actually contains gives -41 dB, so the budget
closes to about 2 dB.

A bed is therefore not "the bright one" because of its contrast alone. It is
bright at the antenna only until the path catches up with it, and by 500 m it
already has.

At 0.30 m cells that rock is sampled at 4.9 nodes per wavelength, under the five
this package otherwise works to. What is under-resolved there is the transmitted
wave, which the PML absorbs and which never returns; the reflection is set by
the impedance step and by the incident wave, which the ice resolves at 6.7.
`tests/test_fdtd.py` checks that rather than asserting it - at example 1's own
spacing a strong and a weak contrast reflect in the ratio of their Fresnel
coefficients to within 8 percent, so the coarse rock costs about 0.6 dB on the
bed and nothing on the layering.

The movie has two panels. The left is the total electric field. The right is the
trace the antenna records - receiver co-located with the transmitter - filling
in as the wave travels, shown as returned power in dB referenced to the transmit
pulse, with dotted lines at the two-way time of every layer put into the model
and a dashed line at the bed. The peaks land on the lines.

The recorded trace and the section are plotted against two-way time. Layer
positions are converted to time by integrating the actual slowness profile
(`IceColumn.two_way_time`); labelling a depth axis with the solid-ice velocity
would misplace every shallow reflector, because firn is roughly 25 percent
faster. Each layer time is measured from the antenna, which is 1 m above the
surface, and carries the offset of the wavelet's own envelope peak
(`envelope_peak_time`); without those two corrections every mark would sit
~16 ns early. Wherever a time axis sits beside a depth axis - the movie's trace panel
beside its wavefield panel - it is scaled through that same profile, so a
reflector at depth `z` on the left lands at the same height on the right.
`trace.png` and the section are cropped at 5.9 us, past which the trace is
showing the wave interacting with the bottom of the model rather than with the
ice.

### 2. Fabric birefringence

Ice with a horizontal fabric is birefringent: the two eigenpolarisations travel
at slightly different speeds and the delay between them accumulates with depth.
That delay is what a polarimetric interferogram measures.

Both eigenpolarisations are propagated with the out-of-plane solver, each given
its own permittivity component. For a fabric column that varies only with depth
this is exact - a wave polarised along x propagating in the y-z plane obeys the
same scalar equation as a wave polarised along y in the x-z plane, with `eps_xx`
in place of `eps_yy` - and it puts both runs on the identical grid with the same
source and receiver nodes, so numerical dispersion cancels in their difference.

The movie shows the two polarisations side by side in essentially the strongest
horizontal fabric ice can have, `lam ~ (0.97, 0.01, 0.02)`. The grain anisotropy
`deps = 0.034` caps the effect, so this is the largest speed difference
physically available: **0.51 percent**. The model runs 800 m deep and narrow
(+/-45 m; nothing here needs width, and the first Fresnel zone at 800 m is only
~30 m across), which is what buys the separation: by 780 m the slow
polarisation is **3.9 m** behind the fast one. The movie keeps 240 frames
regardless of record length, so a deeper run plays back in the same ten seconds
with the wave moving faster through it.

The view scrolls down with the measured wavefront, and a dashed line marks the
fast-axis wavefront in *both* panels, so the lag is read against a fixed
reference rather than by eye: the fast panel runs several metres past the line
while the slow panel stops at it. Once the wave leaves the model the view pulls
back to the whole domain. The third panel is the trace the receiver records,
both polarisations in dB, filling in as the run proceeds.

Ridge A's own `dlam` is roughly twenty times smaller, still measurable but not
something you can watch on a screen, so the movie uses the strong case and the
numbers below use Ridge A.

The figure carries the quantitative chain for Ridge A:

1. the inverted `dlam = lam_perp - lam_par` profile (Open Polar Radar frame
   20250108_02_009),
2. FDTD-measured delay against the traveltime model, for both Ridge A's own
   fabric and the strong one - the solver tracks the model to within 2 percent
   across a 31x range of accumulated delay (719.4 ps measured against
   705.1 ps modelled for Ridge A at 760 m; 22073 ps against 21677 ps for the
   strong fabric),
3. a synthetic interferogram for the full 1850 m column at 195 MHz,
4. the fringes inverted back to `dlam` and compared with the input.

`received.png` carries eight monostatic traces in two rows, as returned power in
dB on one shared reference, each with an inset magnifying the deepest layer
return in a common time window. The top row is the two fabrics by the two
**eigenpolarisations**; the bottom row is the same two fabrics for a **45 degree
launch**, co-pol and cross-pol, which is what a real survey usually transmits. A
45 degree launch is not an eigenmode - it is the two eigenmodes at equal
amplitude - so no new simulation is needed: with the eigenaxes fixed in depth,
as they are here, the two modes propagate independently and the superposition is
exact. Every marked arrival is timed from the antenna, which is 3 m down in the
firn, and carries the wavelet's own offset, so the marks land on the peaks rather
than ~16 ns off them. The figure separates the two things the fabric could do to
a measurement:

* **speed**: in the strong fabric the 700 m return arrives about **42 ns** later
  on the slow axis than the fast one, against 39.7 ns from the traveltime model;
  in Ridge A's the same return is a couple of ns late.
* **amplitude**: it depends entirely on how the antenna is oriented. Along the
  eigenpolarisations there is nothing resolvable - the layers are isotropic, so
  both modes see the same reflector, the conductivity is scalar so neither is
  absorbed preferentially, and the figure prints the measured difference at the
  deepest nadir receiver. Off the eigenaxes the signature appears: at 45 degrees
  the two modes recombine at the receiver with whatever differential phase they
  have accumulated, so co-pol adds them and cross-pol differences them and both
  beat through a null every half cycle of that phase. The bottom row is the same
  delay showing up as amplitude.

The delay is read off the reflection only after the window is detrended and
tapered (`radarwave.polarimetry.isolate_arrival`). These echoes stand about
10 dB above the wake the transmit pulse leaves behind, so over a window several
cycles wide the wake still holds most of the energy - and it has travelled
nowhere, so correlating the raw window reports a 42 ns split as 3 ns.

So birefringence is a traveltime effect at heart: with isotropic reflectors
(these are acidity layers, which both polarisations see identically) neither
eigenpolarisation comes back stronger than the other. It nonetheless reaches the
amplitude of a real measurement by two routes. Off the eigenaxes it does so
through the modes recombining, which is the bottom row above and what
polarimetric sounding lives on. And the reflector itself becomes
polarisation-dependent when it *is* a fabric contrast, which is example 3.

Step 3 is not FDTD. Ridge A is ~1850 m thick and MCoRDS runs near 195 MHz, so
the column is roughly 4000 wavelengths deep - a 2-D grid that size is out of
reach. The fringes are synthesised from the same dielectric model instead and
pushed through the same processing chain applied to the real frames.

With the Ridge A profile and the `+ptt` constants, the whole column gives about
5.6 ns of two-way delay, which is **1.1 fringes at 195 MHz**. Worth checking
against the observed interferogram.

### 3. Dipping fabric transition

A strong fabric change across a plane dipping at 35 degrees, 175 m below the
surface, blended over 1.2 m rather than stepped - real fabric evolves with
strain, and the blending is priced: it costs the return about 25.5 dB against
the step-interface numbers quoted below (`TRANSITION_WIDTH` in the example
carries the cost curve). Two consequences, and they are the point of the
example:

**The energy that returns is not from beneath the antenna.** A monostatic
antenna records a specular return only from where the interface normal points
back at it. For a plane dipping at `delta` at depth `h` below the antenna, that
place is offset sideways and the event is recorded at slant range `h cos(delta)`
- shallower than the interface is. Here the return comes from 83 m to the side,
at 117 m depth, and images at a range of 144 m rather than the 176 m of air and
ice directly beneath the antenna. Across a profile the feature
images with `tan(dip_apparent) = sin(dip_true)`: 30 degrees for a 35 degree
interface.

The section's depth axis is two-way time inverted through the actual slowness
profile (`IceColumn.depth_from_two_way_time`), not divided by a single
solid-ice velocity - which is 25 percent too slow through firn and would put
the imaged event ~15 percent shallow while the predicted-position overlay is
drawn in true metres. The rows are resampled onto an even depth grid before
display, since that conversion is not linear and `imshow` can only stretch an
image linearly between its extent limits.

**It is a polarisation-dependent reflector.** A fabric contrast is a contrast in
the permittivity *tensor*, so the two eigenpolarisations see different jumps.
The transition is a vertical single maximum over a horizontal one,
`lam = (0.115, 0.085, 0.80)` above and `(0.860, 0.060, 0.08)` below, so
`dlam` goes 0.03 -> 0.80 while `lam_y` barely moves. The 89 degree
polarisation therefore sees a strong contrast (**-54 dB**) and the 179 degree
one almost none (-83 dB): the same interface is bright in one polarisation and
nearly invisible in the other, by **29 dB**. A density or acidity layer cannot
do that.

Note that `dlam = 0.8` fixes which polarisation is bright. The horizontal budget
`h = 1 - lam_z` has to be at least 0.8, so `lam_z` must collapse across the
transition and `lam_x` necessarily takes the large jump; the example works out
the bright axis from the permittivities rather than assuming it.

The scene also carries ordinary conformable meteoric layering - all the layers
share one undulation shape with the amplitude growing downwards - so the fabric
transition reads as a discordant feature cutting across a conformable
background, which is what the radargram looks like. One gap in that background
is deliberate: a 30 m band of layers around the depth whose two-way time
matches the transition's arrival is left clear, so the faint fabric echo stands
alone in its window instead of under a layer wavelet.

The movie has the wave propagating on the left and, on the right, the trace the
surface receiver is building up: returned power in dB across, two-way time down,
filling in as the simulation runs, with the predicted arrival of the fabric
transition marked. Power is referenced to the transmit pulse, which is how a
processed sounding is displayed and what keeps a 100 dB range readable.

One thing that falls out of the model and is worth saying out loud: a fabric
contrast is bounded. The grain anisotropy is `deps = 0.034`, so even a complete
change of fabric gives at most `r = deps / (4 eps) = 2.7e-3`, i.e. **about
-51 dB**. Density layering routinely reaches -35 to -45 dB. A fabric reflector
is therefore intrinsically 10-20 dB weaker than the layering around it, before
any gradational smoothing, which is a large part of why the deep feature in the
real radargram is faint enough to carry a question mark.

Both predictions come out of the simulation: the return arrives at 1.439 us
against 1.456 us predicted, 1.2 percent apart, and the polarisation contrast is
29.4 dB measured against 29.4 dB predicted.

The predicted arrival is timed along the specular ray through the firn-corrected
velocity profile, from the antenna a metre above the surface, and carries the
wavelet's envelope-peak offset, so it is the same quantity as the measured
envelope peak - a single solid-ice velocity applied to the ice part of the same
ray would give 1.716 us instead.

The remaining 17 ns is not accounted for. Two candidates: the prediction is a
straight-ray idealisation of a path through a velocity gradient, which really
bends, and the measurement is an envelope peak off a dipping plane whose first
Fresnel zone is metres across, so the return integrates over a band of it rather
than a point. Nothing in this repository establishes which of those dominates,
or even which way the ray-bending term pushes the prediction.

### 4. Banded fabric

Example 3 ends at a ceiling: a single fabric interface can never beat
`r = deps / (4 eps)`, about -51 dB, and gradational smoothing only takes it
further down. This example shows the one mechanism that beats the ceiling -
coherent stacking. Strain banding, alternating fabric bands of the kind ice
cores actually show, turns N weak interfaces into a Bragg mirror whose primary
reflections add in phase at `f = v / (2 d)` for band period `d`. Twenty bands
of the same contrast as example 3's single transition lift the response by
roughly `20 log10(2N)` at resonance, from about -54 dB to about -22 dB,
competitive with the brightest density layering anywhere in the column.

The package carries example 3's fabrics and twin-difference processing, but its
own geometry: deeper and gentler, dipping at 12 degrees and crossing 350 m below
the surface at the centre of the domain. Dip and depth move together, because
the specular point sits `h sin(delta) cos(delta)` updip of the antenna - at
example 3's 35 degrees a 350 m package would return from 165 m off-axis, outside
any affordable domain, while at 12 degrees it returns from about 71 m. The
transmitter is offset 55 m right of centre, which stretches that ray across the
frame and puts the specular point - 19 m left of centre, at 346 m depth - well
clear of the domain edge, with the return arriving past 4 us and clear of the
shallow layering. `geometry.png` draws the scene, the specular ray and the
permittivity profile the wave meets, with an inset on the bands themselves:
1.4 m apart, and invisible at domain scale.

The package is not alone in the column, and the other half of the picture is
what it is read against. Conformable meteoric layering runs from the firn to the
floor of the domain: flat horizons whose specular return comes straight back
from nadir, against one tilted package whose return comes from 71 m off to the
side. That is what a radargram is normally made of, so the record has to carry
both. Below firn close-off the horizons are acidity - conductivity - contrasts,
because density banding contributes essentially nothing there, and this example
asks for stronger ones than the shallower examples need: 4 to 9 times the
background conductivity, spaced 16-26 m, which gives 21 horizons between
close-off and the floor. Nothing is excluded. Example 3 clears a band of layers
around its event because there the transition and the layering come back within
a few dB of each other, whereas here the package returns **+18.5 dB** above the
brightest deep horizon (-73.5 against -92.0 dB in the record), so a coincident
horizon cannot hide it - and a gap cut in the stratigraphy exactly where the
answer is would be the first thing to distrust. The stratigraphy does put one
horizon at the package's own two-way time: 346.4 m under the antenna, arriving
81.7 ns before it, unresolved inside the compressed wavelet. It comes back
-107.2 dB, 33.7 dB below the package, and moves the measured package level by
0.1 dB. Each side of the margin is measured on the record that holds it alone -
the package on the twin difference, the horizons on the package-free twin's own
gather - so the search for the brightest horizon runs to the floor of the record
and includes that coincident one rather than stopping short of the event. The movie draws every horizon
over the wavefield, with the package outline heavy over them, and marks each
horizon's two-way time on the trace, so the record reads as a train of nadir
echoes with a single arrival that belongs to none of them.

The resonance is the fingerprint. The same 20-band package is sounded at two
pulse frequencies: 60 MHz, where the banding is tuned, and 45 MHz, a
quarter-octave below. On resonance the return is bright; off it, it
collapses. The measured on/off contrast is **+19.5 dB** against +20.7 dB from
the transfer matrix - a lower bound, since the off-resonance window sits on
the deep-layer transmission-residual floor rather than on silence, and
brightening the deep horizons raises that floor rather than the signal. A
reflector that appears in one frequency band and vanishes in another is banded
fabric; nothing else in the reflectivity budget does that.

Two presentation choices are deliberate. The transmitted wavelet is a
narrowband (12 percent) Gabor rather than the broadband impulse the other
examples use: the stack resonance is only about 1/N wide, so an impulse
excites it from any centre frequency through its spectral skirt and the
contrast collapses to a couple of dB - and a band-limited waveform is what a
real chirped system delivers after pulse compression anyway (MCoRDS is about
15 percent). And the received traces, in both the figure and the movie's
trace panel, are shown pulse-compressed - correlated with the transmit
wavelet - so every echo wears the zero-phase autocorrelation shape a
processed radar product shows; on these noise-free traces that is a display
transform, not an SNR gain.

The figures carry no words beyond their labels: no titles, no captions, no
callouts, and every axis label is two words plus a unit wherever the quantity
has one - relative permittivity is a ratio and gets none. What a caption would
have said is here and in the module docstring instead. One thing the panel
therefore cannot show is the nadir horizons' own reflections - they come back
around 70 dB under the incident field, which is where the two-dimensional line
source's wake also sits, so every opacity that reveals a horizon reveals the
wake with it and the frame turns to speckle. The wavefield panel carries the
geometry, and the trace panel, which has the dynamic range for it, carries the
returns.

### 5. An ApRES time series, one visit a year

Examples 1-4 ask what is down there. Examples 5 and 6 ask what the ice is
*doing*. One chirp goes into a firn/ice column carrying ordinary meteoric
stratigraphy and the record is kept; the column is deformed by a year of
`radarwave.VerticalStrain` and the identical chirp is transmitted again. Five
visits, four years. The difference between consecutive range-compressed
records - a phase per echo, and so an apparent motion per layer - is the
measurement, and it is good to under a centimetre out of motions of tens of
centimetres.

**Why a series and not a pair.** A pair shows that the phase carries the motion.
A series shows the two things a pair cannot. The motion *accumulates*: by year
four the shallowest markers have moved close to two metres, several resolution
cells and six or seven fringes, so following one marker means tracking it
through steps that each move it more than a resolution cell. And the phase wraps
repeatedly on the way, so the measurement only holds together if every wrap is
accounted for - which is what a real ApRES deployment, recording for a season
and processed pair by pair, has to do. Consecutive pairs rather than everything
against year zero, for the same reason: against year zero the interferogram
would have decorrelated and the unwrap would have nothing to hold on to.

**The deformation.** Two terms move a marker in the depth-below-surface
coordinate a surface-referenced radar works in, and they pull opposite ways.
*Burial*: snow accumulates and the firn beneath compacts, so a marker sinks at
`b / rho_rel(z)` per year - 0.57 m/yr at the surface, 0.22 m/yr in solid ice,
entirely a firn effect and strongly depth-dependent. *Dynamic thinning*: a
vertical strain rate shortens the column above the marker by `eps_zz * z` per
year, lifting it back towards the surface, linear in depth and the term
glaciology wants. At 0.20 m/yr accumulation and `eps_zz = -4e-3` /yr they
balance at about 72 m: shallow markers sink half a metre over the year, deep
ones rise a third of one, and the apparent motion changes sign in between.
That sign change is the signature - a timing error, a repositioned antenna or a
tide lifting the whole column cannot imitate it, because none of them is a
function of depth.

The firn column is treated as being in steady state, so the mean density field,
the wave speed and the snow surface are identical at both epochs and only the
material markers move through them. That is what makes the two records
comparable: every difference between them is deformation. A transient firn
column would change the velocity structure too, and a real repeat sounding
cannot separate that from deformation without an independent constraint;
nothing here models it.

**Why a chirp.** This is the first example that cannot use an impulse. The
displacement is read off `-2 pi f_c dtau`, and that expression needs an `f_c`;
an impulse whose instantaneous frequency wanders across an octave does not have
one. The source is a linear-FM chirp across the real ApRES band, 200-400 MHz,
and two numbers follow from that band and set everything the example shows:
the range resolution `c / (2 B sqrt(eps))` = 0.42 m (0.70 m once the Blackman
band window is applied), and the fringe spacing `c / (2 f_c sqrt(eps))` = 28.1
cm, which is how far a layer must move to turn the phase once.

The chirp simulated is 100 ns rather than the instrument's 1 s: deramp on
receive needs a sweep far longer than the two-way time to the deepest target,
and an FDTD record is microseconds long. The *band* is the instrument's, so the
compressed wavelet, the resolution and the sidelobes are too. What is not
reproduced is the time-bandwidth product, and with it the SNR gain the long
sweep exists for - and these traces carry no noise for it to improve.
`radarwave/waveform.py` carries the deramp chain as well as the matched filter,
and the tests check that the two return the same range profile and the same
differential phase rather than asserting it.

**How the motion is recovered**, per consecutive pair, which is the ApRES chain
and what `+vdef` (`~/projects/radar_velocity`) does to the EAGER repeat passes:
find every echo in the earlier record, cross-correlate the later envelope
against it, unwrap the interferometric phase along depth, fix the one remaining
whole-fringe ambiguity from the envelope estimates, read the range change off
the phase, and chain echoes from step to step into marker trajectories. The
envelope's job is that ambiguity and nothing else. It has to be right to half a
fringe to do it - 14 cm at 300 MHz, 5.6 cm at 750 - and both examples print how
close it came, because that margin is what fails first when a band is pushed up.
Tracking drops a marker whose predicted position has no echo near it rather than
matching it to the nearest thing available: chaining onto the wrong reflector
does not look like an error afterwards, it looks like a layer that moved several
metres in a year.

**What the two runs measured.** Both sound the identical column and
deformation; they differ only in the band.

| | ApRES, 200-400 MHz | accum3, 600-900 MHz |
| --- | --- | --- |
| domain sounded | 140 m | 60 m |
| echoes found | 22, over 3-94 m | 10, over 4-39 m |
| markers followed through all five visits | 20 | 6 |
| phase turned per year, across the record | 2.1 | 1.9 |
| apparent motion per year-step | **0.69 cm rms** (85 measurements) | **0.84 cm rms** (38) |
| cumulative motion after 4 years | 2.70 cm rms, out of 24-205 cm | 2.42 cm rms, out of 90-207 cm |
| envelope estimate | 2.27 cm rms, worst 6.11 | 3.77 cm rms, worst 10.75 |
| envelope estimates outside the half-fringe | **0 of 85** (bar 14.0 cm) | **5 of 38** (bar 5.6 cm) |

The last row is the trade, measured. accum3's fringe is 2.5x tighter, so its
whole-fringe ambiguity is 2.5x harder to resolve, and individual envelope
estimates *do* cross the bar. What saves it is that the fringe integer is chosen
once per profile from the **median** of the per-echo estimates rather than per
echo - a handful of bad ones then cost nothing, where a biased majority would
cost a whole fringe. That is a design choice worth knowing about before pushing
a repeat-pass measurement to a higher band.

**The per-step errors do not average down.** Each year-step is recovered to
about 0.7 cm rms, but the cumulative motion after four steps is out by 2.7 cm,
not the 1.4 cm independent errors would give - it is very close to the 2.7 cm a
*per-reflector* offset repeating at every step produces (accum3: 2.42 cm
measured, 1.68 independent, 3.37 fully correlated). Each
echo has its own small persistent error, set by its own local geometry and where
it happens to fall on the grid, and that error is the same every year. Worth
knowing before quoting a precision for a season-long record: the scatter between
reflectors is what averages down, the offset of any one of them does not.

**No movie**, deliberately, unlike the other four. The medium is identical at
every epoch, so consecutive wavefields differ by less than a metre in 140, which
is under a pixel; there is nothing to watch. The four figures are the pulse
itself (`pulse.png`), the records and the phase between them (`record.png`), the
recovered apparent motion against the imposed profile at every step
(`motion.png`), and the marker trajectories beside the wrapping phase the radar
actually reads (`series.png`).

Three things had to be got right before any of this measured anything, and each
is worth more than its line count:

* **Sub-cell layer positions.** A hard-edged layer can only sit on a grid node,
  so displacing it by less than a cell displaces it by zero or by a whole cell.
  At 3.5 cm cells that quantisation is up to a radian of phase, larger than
  much of the signal. `Layer.edge_width` ramps the edges over a couple of cells
  so the sampled profile's centroid follows the requested depth continuously.
* **A layerless twin.** The layer echoes run 85-92 dB below the transmit pulse,
  and the wake a 2-D line source leaves behind itself sits near -95 dB, so an
  echo stands only 5-10 dB above something that did not move. Subtracting a
  no-layer twin of the same column from both epochs cut the phase error from
  5.3 cm to 0.7 cm on a 42 m test column. It is one twin for both epochs,
  because the background is exactly what does not change.
* **The coarse estimator.** `subsample_lag` fits a parabola to the correlation
  peak, which is right for the zero-mean correlations it was written for and
  wrong for a one-signed compressed envelope: the mean removal digs a pedestal
  and pulls the vertex towards zero lag by several centimetres, a third of an
  ApRES fringe and most of an accum3 one. `upsampled_lag` interpolates the
  correlation instead, which is what `vdef.coalignPair` does. The window
  matters as much as the estimator - it has to be several times the compressed
  wavelet or the shifted copy is truncated asymmetrically, and shorter than the
  reflector spacing or it measures two layers at once.
* **Telling an echo from structure on one.** A compressed echo is a wavelet, not
  a spike, and peak-finding on amplitude alone returns half again to twice as
  many "echoes" as there are layers - each extra one carrying its parent's phase
  rather than a phase of its own, which is what turns a one-centimetre
  measurement into a five-centimetre one at 750 MHz. A peak is kept only if it
  is within 10 dB of the strongest sample within six resolution cells. That is
  conservative on purpose: it throws away a genuine reflector that happens to be
  10 dB weaker than a close neighbour, which costs one point on a profile of
  twenty, where a retained sidelobe puts a *wrong* point on it and nothing
  downstream can tell which it was.

### 6. The same series in the EAGER accum3 band

The same column, the same deformation, the same five visits and the same
processing, all imported from example 5, sounded with the CReSIS/OPR
accumulation radar instead: 600-900 MHz in survey mode, from
`default_radar_params_2022_Antarctica_Ground_accum.m`, the configuration run on
the 2022_Antarctica_Ground traverse at Windless Bight whose repeat passes the
EAGER vertical-deformation product is built from. The only thing that differs
between the two examples is the band, which is what makes the pair a comparison
of waveforms rather than of two experiments that happen to resemble each other.

| | ApRES | accum3 |
| --- | --- | --- |
| band | 200-400 MHz | 600-900 MHz |
| range resolution in ice | 0.42 m (0.70 m windowed) | 0.28 m (0.46 m windowed) |
| one turn of phase | 28.1 cm | 11.2 cm |

accum3 separates two reflectors half again as finely and reads a given motion
with two and a half times the phase - and wraps two and a half times as often
doing it. Both show up here. The finer resolution splits some firn bands into
their top and bottom interfaces, which ApRES sees as a single echo. The tighter
fringe means the envelope estimate that resolves the whole-fringe ambiguity has
to stay inside 5.6 cm rather than ApRES's 14 cm. More fringes is more signal only
for as long as they can still be counted, and that is the trade a campaign is
making when it chooses a band rather than maximising one.

The domain is shallower and has to be: resolving 900 MHz in ice needs a 1.6 cm
cell against ApRES's 3.5 cm, so the same depth would cost about five times the
cell-updates. The instrument does not reach that far anyway - an accumulation
radar is a firn instrument and the EAGER strain result is quoted over the top
100 m - so this example sounds 60 m rather than 140. The sign change at 72 m is
below that, so every marker here is sinking: what accum3 sees is the top,
steepest part of the same curve, in finer detail.

## Package layout

```
radarwave/
  fdtd.py         O(2,4) FDTD with convolutional PML; TM (Ey) and TE (Ex,Ez) modes
  grid.py         property grids, discretisation criteria, padding, resampling
  ice.py          firn density, bubbles, fabric -> permittivity tensor
  scenes.py       building 2-D ice models: layers, dipping surfaces, fabric domains
  polarimetry.py  interferograms, delays, fabric inversion
  sources.py      source wavelets
  waveform.py     chirped radar systems, range compression, FMCW deramp
  deform.py       Lagrangian vertical strain between two repeat soundings
  viz.py          talk-quality figures and MP4 wavefield movies
  constants.py    physical constants
```

### Dielectric model

`radarwave/ice.py` is a depth-parameterised port of the polarimetric traveltime
toolbox used to invert Open Polar Radar data (`fabric_anisotropy/+ptt`),
following Rathmann (2026) and Fujita et al. (2000). It uses the same constants
(`eps_bar = 3.171`, `deps = 0.034`), the same Herron-Langway firn profile, the
same bubble depolarisation tensor and the same Maxwell-Garnett mixing. A
synthetic radargram made here can therefore be pushed through the same inversion
that is applied to the real frames - which is what example 2 does.

### Anisotropy

`PropertyGrid` carries permittivity and conductivity per principal axis
(`xx`, `yy`, `zz`). The fabric principal axes are assumed aligned with the model
axes, which covers vertical-single-maximum, girdle and horizontally anisotropic
ice. Off-diagonal terms - a fabric whose symmetry axis is tilted *within* the
model plane - are not supported.

### Resolution

Numerical dispersion sets how fine the grid has to be. For arrival times, six
nodes per wavelength is enough. For the birefringent delay it is not: the delay
is a fraction of a time step and the dispersion error differs slightly between
the two runs because their wave speeds differ. Measured against the traveltime
model, the error is about +20% at 5.6 nodes per wavelength, +3.5% at 8, +0.6% at
11 and +0.3% at 16. The examples run at 9-14; `--quick` runs at 5.6 and is for
checking the pipeline, not for numbers.

## Changes from the original code

The Python in this repository previously did not run: `model.py` imported
`scipy.interpolate.interp2d`, removed in SciPy 1.14, and `simulation.py` called
a function that does not exist (`blackharriswave`, at three of its four call
sites) on a `crosshole_model.mat` that is not in the repository. `model.py` is
now a shim over the package. `simulation.py` is untouched and still broken; it
is superseded by `examples/`, which is what these three simulations are driven
from. Beyond that:

* **The absorbing boundary was reflecting at -23.5 dB** and, tellingly, did not
  improve when the layer was made thicker. The cause was `kappa_max = 5` real
  coordinate stretching, which divides the damping conductivity by up to 5.9 and
  throws away most of the absorption. With `kappa_max = 1` the boundary reaches
  -45 dB at `npml = 10` and -51 dB at `npml = 20`, and it scales with thickness
  again. The original MATLAB-derived solver reproduces the -23.5 dB figure
  exactly, so this was inherited rather than introduced. It matters because the
  fabric reflection in example 3 is near -80 dB: -54 dB at normal incidence for
  the bright eigenpolarisation, less the 25.5 dB its 1.2 m gradational transition
  costs.
* Update coefficients are pre-sampled onto each field component's own grid, so
  the time loop is slice arithmetic rather than `np.ix_` fancy indexing - about
  an order of magnitude faster, with no per-step allocation.
* Per-axis permittivity and conductivity, which is what makes the fabric
  examples possible.
* Fixed the receiver-gather and output-time indexing (the earlier port wrote the
  first sample to index -1 and stored shot 0 in the last column), and built the
  field coordinate vectors from integer strides rather than `np.arange` on
  floats, which silently dropped a node.
* Movies are written straight to MP4 by mutating one Matplotlib figure and
  grabbing the canvas - no intermediate PNG sequence.

`model.py` is kept as a thin compatibility shim so `crevasse.ipynb` still works.

## Validation

`pytest` covers, among other things:

* propagation speed against `c / sqrt(eps)`, and that the error falls with
  refinement;
* reflection amplitude off a planar contrast against the Fresnel coefficient,
  both absolutely and as a ratio between two contrasts;
* the birefringent split between polarisations against `2 d (s_x - s_y)`,
  including its convergence with grid refinement;
* that TM responds to `eps_yy` and is blind to `eps_xx`, and vice versa;
* boundary reflection below -40 dB, and that it improves with layer thickness;
* the full polarimetric chain: synthesise fringes from a fabric profile, process
  them, and recover the profile;
* that `--bare` reaches inset axes and figure-level legends while leaving the
  axis labels, tick numbers and colourbar text a slide still needs;
* that a snapshot cache written from one model is refused by another, naming the
  parameter that moved;
* the chirped chain examples 5 and 6 run on: that a compressed echo lands at its
  own two-way time with no wavelet offset, that the phase between two compressed
  profiles is exactly `-2 pi f_c dtau`, that the FMCW deramp chain and the
  matched filter return the same delay and the same differential phase, and that
  the whole repeat-pass recovery - synthesise a pair from a known deformation,
  find the echoes, resolve the fringe, read the phase - returns the deformation
  it was given across more than a full fringe of motion;
* that a ramped layer edge tracks a sub-cell displacement where a hard-edged one
  quantises, and that `edge_width = 0` reproduces the old hard mask node for
  node;
* that `upsampled_lag` is unbiased on a compressed envelope where the parabolic
  `subsample_lag` is not;
* that a sampled deformation trajectory is the same path as integrating straight
  to each time, and that marker tracking chains reflectors across a series and
  drops one it cannot match rather than chaining onto its neighbour.

Beyond the test suite, `validation/` holds cross-code checks against tools the
package deliberately does not depend on: `thin_layer_check.py` scores the
solver against the exact transfer-matrix response of a thin layer, the
`*gprmax*` scripts run the same models through a local gprMax build (its
python and repo paths are passed on the command line; gprMax is never
imported), and `emmodel_export.py` / `emmodel_driver.m` bridge to the
CReSIS/OPR `em_model` MATLAB library, whose driver runs wherever that toolbox
lives. Each script's docstring states what it validates and what it cannot;
their figures land in `figures/validation/`.

## References

Irving, J. and Knight, R. (2006). Numerical modeling of ground-penetrating radar
in 2-D using MATLAB. *Computers and Geosciences* 32, 1247-1258.

Roden, J. A. and Gedney, S. D. (2000). Convolutional PML (CPML): an efficient
FDTD implementation of the CFS-PML for arbitrary media. *Microwave and Optical
Technology Letters* 27, 334-339.

Fujita, S., Matsuoka, T., Ishida, T., Matsuoka, K. and Mae, S. (2000). A summary
of the complex dielectric permittivity of ice in the megahertz range and its
applications for radar sounding of polar ice sheets.

Rathmann, N. (2026). Polarimetric traveltime model for anisotropic ice.

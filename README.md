# radar-waveform-model

Two-dimensional FDTD modelling of radar waves in polar ice.

The solver started as a Python port of the ground-penetrating-radar FDTD code of
Irving and Knight (2006) and has been extended to handle the **anisotropic**,
depth-varying dielectric structure of an ice sheet. That extension is what lets
it model crystal-orientation fabric, and therefore the three things a
quad-polarimetric ice sounder actually sees:

| example | what it shows |
| --- | --- |
| `examples/ex01_meteoric_layers.py` | a wave propagating into ice and reflecting off meteoric internal layering |
| `examples/ex02_fabric_birefringence.py` | two polarisations separating in ice with a preferred orientation fabric |
| `examples/ex03_dipping_fabric_transition.py` | a fabric transition that is not at nadir but still returns energy to the antenna |

Each example writes an MP4 of the propagating wavefield plus supporting figures
into `figures/`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # ~57 tests, a few minutes
```

## The three examples

```bash
python examples/ex01_meteoric_layers.py --radargram --processes 10
python examples/ex02_fabric_birefringence.py
python examples/ex03_dipping_fabric_transition.py --radargram --processes 10
```

Add `--quick` to any of them for a small, fast version while iterating on the
model, and `--render-only` (examples 2 and 3) to re-render the movie from cached
snapshots without repeating the simulation.

### 1. Meteoric layering

A 60 MHz pulse from a surface antenna into a firn/ice column carrying the two
kinds of contrast that make up meteoric stratigraphy: density (permittivity)
banding in the firn, and volcanic acid (conductivity) horizons in the solid ice
below. The layers undulate, with the amplitude growing downwards.

The movie has two panels. The left is the total electric field. The right is the
trace the antenna records - receiver co-located with the transmitter - filling
in as the wave travels, shown as returned power in dB referenced to the transmit
pulse, with dotted lines at the two-way time of every layer put into the model.
The peaks land on the lines.

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

`received.png` carries the four monostatic traces - two fabrics by two
eigenpolarisations - as returned power in dB on one shared reference, each with
an inset magnifying the deepest layer return in a common time window. Every
marked arrival is timed from the antenna, which is 3 m down in the firn, and
carries the wavelet's own offset, so the marks land on the peaks rather than
~16 ns off them. It separates the two things the fabric could do to a
measurement:

* **speed**: in the strong fabric the 700 m return arrives about **42 ns** later
  on the slow axis than the fast one, against 39.7 ns from the traveltime model;
  in Ridge A's the same return is a couple of ns late.
* **amplitude**: nothing resolvable. Both eigenpolarisations see the same
  isotropic reflectors and the same conductivity, and the figure prints the
  measured difference between them at the deepest nadir receiver.

The delay is read off the reflection only after the window is detrended and
tapered (`radarwave.polarimetry.isolate_arrival`). These echoes stand about
10 dB above the wake the transmit pulse leaves behind, so over a window several
cycles wide the wake still holds most of the energy - and it has travelled
nowhere, so correlating the raw window reports a 42 ns split as 3 ns.

So birefringence is a traveltime effect, not an amplitude one, as long as the
reflectors themselves are isotropic (these are acidity layers, which both
polarisations see identically). Amplitude only becomes polarisation-dependent
when the reflector *is* a fabric contrast, which is example 3.

Step 3 is not FDTD. Ridge A is ~1850 m thick and MCoRDS runs near 195 MHz, so
the column is roughly 4000 wavelengths deep - a 2-D grid that size is out of
reach. The fringes are synthesised from the same dielectric model instead and
pushed through the same processing chain applied to the real frames.

With the Ridge A profile and the `+ptt` constants, the whole column gives about
5.6 ns of two-way delay, which is **1.1 fringes at 195 MHz**. Worth checking
against the observed interferogram.

### 3. Dipping fabric transition

A sharp fabric change across a plane dipping at 35 degrees, 175 m below the
surface. Two consequences, and they are the point of the example:

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
one almost none (-84 dB): the same interface is bright in one polarisation and
nearly invisible in the other, by **29 dB**. A density or acidity layer cannot
do that.

Note that `dlam = 0.8` fixes which polarisation is bright. The horizontal budget
`h = 1 - lam_z` has to be at least 0.8, so `lam_z` must collapse across the
transition and `lam_x` necessarily takes the large jump; the example works out
the bright axis from the permittivities rather than assuming it.

The scene also carries ordinary conformable meteoric layering - all the layers
share one undulation shape with the amplitude growing downwards - so the fabric
transition reads as a discordant feature cutting across a conformable
background, which is what the radargram looks like.

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
envelope peak - a single solid-ice velocity would give 1.730 us instead. What is
left is the straight-ray idealisation: the ray is taken as one straight line
along the interface normal, and a firn velocity gradient bends it onto a faster
path, which is part of why the measurement lands on the early side. Only a
refracted two-point ray trace would close the remaining 17 ns.

## Package layout

```
radarwave/
  fdtd.py         O(2,4) FDTD with convolutional PML; TM (Ey) and TE (Ex,Ez) modes
  grid.py         property grids, discretisation criteria, padding, resampling
  ice.py          firn density, bubbles, fabric -> permittivity tensor
  scenes.py       building 2-D ice models: layers, dipping surfaces, fabric domains
  polarimetry.py  interferograms, delays, fabric inversion
  sources.py      source wavelets
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
11 and +0.3% at 16. The examples run at 9-11; `--quick` runs at 5.6 and is for
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
  fabric reflections in example 3 sit near -60 dB.
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
  them, and recover the profile.

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

# v60 long-normal coverage preflight

Keep Hopper-v4 physics/profiles, source/replay/context, beta=-2 and all readout
settings. Fit seed132000 and calibration seed132010 each supply6000 normal
transitions from frozen and updating branches. Separate test132020/132021 each
has frozen-normal plus updating normal/heavy/slippery/high_grav/compound6000-step
cells; updating changed cells switch at3800, inside the fourth episode.
Total16 cells,96000 interactions and72000 optimizer updates.

Compare three sensors on identical held-out trajectories: frozen v57; long
normal calibration of the old dynamics/moments/score scale; and a long-normal
refit of dynamics plus signed-window moments, scale and phase curves. The latter
two share the same calibration data. Fit uses only the fitting split, thresholds
use only calibration, and everything freezes before any test cell. Neither
sensor changes actions/beta. Retain residual/departure history across resets.

Normal frozen and updating first800/full/last1000/last64 coefficients must all
remain within the existing0.10 normal bound. Each changed updating first50 and
last64 contrast against its paired normal control must be positive. Report all
three sensors and contrast magnitudes, not just signs. This is a development
detector preflight, not a performance or independent-source confirmation.

Save full native/replay/RNG state every500 steps; only scalar JSON/logs return.
Do not expand closed-loop training if broader coverage masks real changes.

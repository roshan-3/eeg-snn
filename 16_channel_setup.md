# 16-Channel Ultracortex Mark IV Setup

Hardware-side companion to `stack_validation/montage.py`. Documents which
scalp sites are used, which electrode type belongs at each site, and the
wire-color / pin-assignment convention for the Cyton + Daisy boards.

The channel order here is identical to `CHANNELS_16` in
`stack_validation/montage.py:23-29`. Every saved tensor in this repo is
indexed in this order. Never reorder; only append.

## Board

- OpenBCI Cyton (8-channel) + Daisy module (8 more) = 16 channels total.
- Ganglion (4 channels) is *not* sufficient and cannot be daisy-chained.
- Native sample rate with Daisy attached: 125 Hz. The pipeline resamples
  to 250 Hz, so either 125 Hz or 250 Hz on the board is fine.

## Electrode positions (10-20 system)

```
Fp1, Fp2
F3, Fz, F4
T7, C3, Cz, C4, T8
P3, Pz, P4
O1, Oz, O2
```

All 16 are stock holes in the Mark IV chassis; no drilling required.
T7/T8 are the modern names for the older T3/T4 (same physical location).

## Electrode types per site

Material: Ag/AgCl for all sites (do not mix materials, e.g., gold vs
Ag/AgCl, on the same recording — DC offsets fight the CAR).

| Site | Type | Notes |
|---|---|---|
| Fp1, Fp2 | Flat / smooth dry | Forehead, no hair. |
| F3, Fz, F4 | Short-spike dry | Hairline, usually thinner hair. |
| T7, T8 | Short-spike dry | Above the ear, thinner hair. |
| C3, Cz, C4 | Long-spike "comb" dry | Top of head, thickest hair. |
| P3, Pz, P4 | Long-spike "comb" dry | Crown, dense hair. |
| O1, Oz, O2 | Long-spike "comb" dry | Back of head, worst contact sites. |
| Reference (SRB2) | Ag/AgCl ear-clip | Right earlobe. |
| BIAS | Ag/AgCl ear-clip (same model as reference) | Left earlobe. |

For dense / coarse hair, the stock spikes may not reach scalp at
O1/Oz/O2 (and sometimes Cz). Two options:

1. Wet Ag/AgCl cup electrodes with conductive gel at the worst sites.
   Gives a ~10x impedance drop. Recommended for the eyes-open/closed
   task since its signal lives on O1/Oz/O2/Pz.
2. Longer third-party dry combs (e.g., Florida Research Instruments,
   g.tec) that fit the OpenBCI holders.

Rules:

- Match electrode material across all 16 sites.
- Don't mix wet and dry on adjacent sites; if you gel one occipital
  site, gel the whole occipital row.
- Reference and BIAS clips: identical model.

## Wire colors and channel assignments

OpenBCI does not enforce a wire color standard — the board only cares
which pin a wire lands on. The scheme below uses the standard electronic
resistor color code (10 distinct, hard-to-confuse colors). Cyton uses
the code straight; Daisy uses the same colors with a white stripe so
channel 1 and channel 9 can be told apart at a glance.

### Cyton (channels 1-8, pins N1P-N8P)

| Ch | Site | Wire color | Resistor digit |
|---|---|---|---|
| 1 | Fp1 | Brown | 1 |
| 2 | Fp2 | Red | 2 |
| 3 | F3 | Orange | 3 |
| 4 | Fz | Yellow | 4 |
| 5 | F4 | Green | 5 |
| 6 | T7 | Blue | 6 |
| 7 | C3 | Violet / Purple | 7 |
| 8 | Cz | Gray | 8 |

### Daisy (channels 9-16, pins N1P-N8P on the Daisy module)

| Ch | Site | Wire color | Marker |
|---|---|---|---|
| 9 | C4 | Brown | white stripe |
| 10 | T8 | Red | white stripe |
| 11 | P3 | Orange | white stripe |
| 12 | Pz | Yellow | white stripe |
| 13 | P4 | Green | white stripe |
| 14 | O1 | Blue | white stripe |
| 15 | Oz | Violet / Purple | white stripe |
| 16 | O2 | Gray | white stripe |

### Reference and BIAS

| Function | Board pin | Wire color | Site |
|---|---|---|---|
| Reference | SRB2 (Cyton) | White | Right earlobe clip |
| BIAS / Driven ground | BIAS (Cyton) | Black | Left earlobe clip |

White and black are reserved: if SRB2 lands on a channel pin the whole
recording is wrong, and if BIAS is loose every channel picks up mains
noise. Reserving the two extreme colors makes those mistakes obvious.

## Pre-recording checklist

1. Label both ends of every wire (heat-shrink + Sharpie channel
   number). Color alone fails when one end is under the printed shell.
2. Keep wire lengths roughly equal for paired sites (Fp1/Fp2, F3/F4,
   ...). Length mismatch = capacitance mismatch = CMRR mismatch.
3. Twist the BIAS wire with the reference wire from board to ear
   clips. Reduces 60 Hz pickup on the reference that otherwise
   contaminates every channel.
4. Run the OpenBCI GUI impedance check before each session.
   - Dry electrodes: target < 50 kohm.
   - Gel electrodes: target < 10 kohm.
   The Cyton can run higher but you will see it as gamma-band noise
   that the 5-50 Hz Butterworth filter does not remove.
5. Verify channel mapping with a tap test: tap each electrode and
   confirm the expected channel spikes in the GUI. Catches swapped
   wires before they end up in saved data.
6. For the eyes-open/closed task specifically, prioritize impedance
   at O1, Oz, O2, Pz — these four carry the signal.

## If you use OpenBCI's pre-made ribbon cable

The ribbon that ships with some Mark IV kits has fixed colors. Do not
fight the cable: write down whatever its colors are, lock that mapping
into this file, and keep `CHANNELS_16` unchanged. Only the wiring doc
should ever change to match new hardware.

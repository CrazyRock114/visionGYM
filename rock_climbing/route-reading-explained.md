# How the route is read

1. Sample twelve frames across the clip and segment each with SAM 3.1 on the
   colour prompt.
2. Cluster the instances by IoU, keep a hold seen in a quarter of the samples,
   pixel-vote the survivors into one mask.
3. Drop holds outside the climber's own hull, expanded: they are on the next wall.
4. Number the holds bottom to top, along the route's lean.
5. Track the climber with ViTPose on every frame.
6. Start the clock when both feet clear the floor with a hand on a hold; stop it
   when both wrists hold the top hold.
7. Activate each hold after continuous contact, and in batch mode align every
   clip onto one wall's numbering before comparing.

Code: [`src/holds.py`](src/holds.py), [`src/climb.py`](src/climb.py),
[`src/compare.py`](src/compare.py). Every parameter, and why each default is what
it is: [`config.py`](config.py).

## Why these rules

- **The route is a consensus, not a detection.** SAM's edge moves between looks
  at the same hold, so the outline is the agreement between samples. A hold
  nested inside a better-scoring one is dropped: IoU is blind to containment, and
  a phantom hold shifts every id above it. The prompt also finds holds on the
  next wall over, which the climber's own keypoints are what narrow away.
- **The ankle is not the foot.** COCO-17 stops at the joint and contact is at the
  toes, so `ANKLE_TO_TOE_OFFSET` shifts it down for the inclusion test.
- **Hold numbers belong to the wall, not the climber**, or two runs cannot be
  compared hold for hold. Height alone would let mask noise reorder near-level
  holds, so a `HOLD_NUMBER_BAND` of them is one row, ordered along the lean.
- **Contact has two thresholds, and touch is dwell.** A settled keypoint's wander
  reads as regripping against one boundary, so contact is entered at
  `HOLD_MASK_MARGIN` and held until `HOLD_RELEASE_MARGIN`, both isotropic; and a
  hand passing over a hold has not used it, so activation needs
  `HOLD_DWELL_SECONDS`. The top hold confirms on a tap.
- **The clock starts when both feet leave the floor**, not when they find holds:
  a foot smeared flat against the wall is a legitimate placement. The floor is
  segmented too, its top edge kept per column since the junction is not level in
  frame, and a hand must be on a hold, or a climber walking to the wall qualifies.

## Comparing attempts

A sequence is limb-agnostic: `1 2 5 8 10` is the path up the wall, and that is
what two attempts can be laid side by side. Which limb arrived is kept as
`moves`, the per-limb beta as `per_limb_sequence`, both in `sequence.json`.
**on hold %** is contact time over the climb's duration, so the four do not sum
to 100% and each compares against another climber's same limb.

**Hold 7 has to be the same hold in every clip.** Each clip gets its own SAM
pass, so one that missed a hold produces no gap: every number above it is one too
low and nothing looks wrong. Holds are matched by centroid within
`HOLD_MATCH_DISTANCE` onto the wall the clips agree on, and every clip adopts
that numbering everywhere. Step 7 warns on a differing count, an unmatched hold,
a renumbering, or drift past `HOLD_DRIFT_WARN`.

## If it comes out wrong

Look at `holds.png` first: nearly everything downstream is a consequence of it.

| symptom | knob |
|---|---|
| holds missing from the route | `HOLD_MIN_APPEARANCE` down, `HOLD_SAMPLE_FRAMES` up |
| holds from the next wall over | `HOLD_SPATIAL_MARGIN` down |
| two adjacent holds merged into one | `HOLD_IOU_THRESHOLD` up |
| one hold outlined and numbered twice | `HOLD_NMS_CONTAINMENT` down |
| two same-height holds numbered "backwards" | `HOLD_NUMBER_BAND` up |
| a hold the climber used never lights up | `HOLD_DWELL_SECONDS` down, `HOLD_MASK_MARGIN` up |
| footholds never light up | `ANKLE_TO_TOE_OFFSET` up |
| contact fragments, or a limb keeps a hold it has left | `HOLD_RELEASE_MARGIN` up, down |
| no top-out detected | `FINAL_HOLD_DWELL_SECONDS` down |
| the clock starts early, late, or never | `FLOOR_CLEARANCE`; check the floor line |
| the floor was not found | `FLOOR_PROMPT`: name your gym's floor, e.g. `"blue mat"` |
| one clip's sequence looks shifted | step 7's hold check: it found a different count |
| the whole clip looks washed out | HDR that was not tone-mapped; step 3 says why |

`TONEMAP = "auto"` converts HDR sources and leaves SDR ones alone. It runs on
`libplacebo`, which needs a Vulkan device (`moltenvk` on macOS), and Homebrew's
ffmpeg has none, so check `which ffmpeg` finds the environment's one.

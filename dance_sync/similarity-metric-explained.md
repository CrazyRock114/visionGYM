# How the similarity metric is computed

Two dancers doing the same choreography are never geometrically identical, so
the score measures agreement between them rather than correctness. It has two
halves: **posture** (are they in the same shape) and **timing** (are they moving
together), each calibrated against a baseline measured from the clip itself.

Code: [`src/similarity.py`](src/similarity.py) and
[`src/pictures.py`](src/pictures.py). All parameters: [`config.py`](config.py).

## 1. Turn each pose into angles

- The gateway returns 17 COCO keypoints per dancer per frame.
- Give each dancer their own torso frame: origin at the mid-hip, up toward the
  mid-shoulder.
- Measure the direction of 12 body segments inside that frame: thighs, shins,
  upper arms, forearms, shoulder line, neck and head, plus the torso's own lean
  against vertical.
- Angles rather than joint positions, so differences in height and limb
  proportion cannot register as being out of sync.
- Only the choreographed stretch of the clip is scored, and within it only
  frames where every dancer is fully in shot, since a pose reaching off screen
  is the model guessing.

## 2. Posture: compare the shapes

- For every pair of dancers, take the angular difference of each segment.
- Forgive a few degrees per body part first, because a forearm 20° off is
  nothing while a torso 20° off is a different shape.
- Weight the core (torso, thighs, neck) above the extremities, since that is
  what carries a pose and the arms legitimately vary between people.
- Average the segments into one difference, in degrees.
- Let each frame compare against its partner's best instant within ±2 frames,
  so a one-frame offset is charged to timing rather than to shape.

## 3. Timing: compare the movement

- Differentiate the same angles to get how fast each segment is turning.
- Speed only, not direction, since the direction a limb travels is choreography
  and not timing.
- Average the segments the same way, giving a second difference in degrees per
  second.

## 4. Decide what 0% and 100% mean

- A raw difference in degrees is not a score, so each half gets two anchors
  measured from the clip.
- **0%** is the median difference between the dancers at moments at least two
  seconds apart, which is how alike two unrelated moments of this dance already
  look. Two upright human bodies agree substantially by accident, so without
  this the score reports sync that is not there.
- **100%** is the best agreement these bodies actually reach, taken as the 10th
  percentile of the clip's own differences. Identical is unreachable: three
  bodies at three angles through a 2D projection cannot coincide.
- The score is then just where the difference falls between the two anchors,
  clamped to 0-100.
- The trade is that both anchors come from the clip, so a score says where a
  moment sits within that run and cannot be compared against another video.

## 5. Score the landings

- Choreography is a sequence of shapes the dancers land in, and those landings
  are what an audience reads as tight or loose.
- Find each dancer's arrivals as the moments their limbs stop moving, requiring
  a real dip in speed so that slowing down mid-travel does not count.
- Arrivals from different dancers within about 130 ms are the same landing.
- Compare each dancer at their own arrival, so landing late in the right shape
  costs timing and leaves the shape clean.
- Score a landing half on shape, using the same calibration as above, and half
  on how far apart the arrivals were.

## 6. Combine into one number

- Near a landing, timing comes from the arrival spread; between landings it
  comes from the speed comparison, crossfaded so the trace stays continuous.
- Each pair's frame score is half posture, half timing.
- The frame score is the mean over the pairs, and each dancer's score is the
  mean of the pairs they appear in.
- Dancers are ranked on those means, with a bootstrap to size the uncertainty,
  and share a rank unless one clearly wins. On the reference clip all three tie,
  which is the honest answer.

## What this does not do

- **Relative, not correct:** there is no ground-truth choreography, only
  agreement, so if two dancers drift the same way the third looks wrong.
- **Clip-relative:** it maps a run onto its own range and does not grade it
  against another run.
- **2D:** a real change of facing and a projection artifact are
  indistinguishable.
- **Not musical:** the score never looks at the audio.

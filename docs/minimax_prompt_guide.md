# MiniMax-H3 Prompt Writing Guide (key context)

Condensed from the official guide on the Hugging Face model card:
`MiniMaxAI/MiniMax-H3` → `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`
(a separate `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` covers ref2va; skills at
https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills).

## Task types the guide distinguishes

- **T2VA** — text only, no keyframes
- **I2VA** — first frame supplied (`--image`)
- **FL2VA** — first + last frame supplied (`--image` + `--last_image`)
- **L2VA** — last frame only (`--last_image`), the video *converges onto* it

## Prompt structure

The prompt is **one string** containing three labeled sections, concatenated
in this order (this is exactly how the official examples pass it):

```
integrated_multimodal_description: <visuals, actions, dialogue and diegetic
  audio, in chronological order, organized into [Shot N] blocks>
overall_soundscape: <ambient + physical action sounds, 1-4 sentences>
non_diegetic_music: <background score description, 1-3 sentences>
```

## Keyframe alignment (I2VA / FL2VA / L2VA)

State explicitly where each picture lands on the timeline, at the top of the
description, e.g. for a ~5-second FL2VA clip:

> Picture 1 aligns with the 0.00-second mark; Picture 2 aligns with the
> 5.17-second mark.

Then have the action *begin* in the position established by Picture 1 and
*settle into* the pose/composition established by Picture 2. For L2VA, anchor
only the final frame at the video's duration and describe how the scene
converges onto it. (Duration = num_frames / 24; e.g. 124 frames ≈ 5.17 s.)

## Shots and camera motion

- First shot has no timestamp; later shots open with
  `At 00:XX.XXX, the camera cuts to...`. Prefer "cuts to" unless the user
  asked for dissolves/fades.
- Camera motion needs **three dimensions**, woven into natural sentences:
  **type** (push in, pull out, pan, tilt, truck, zoom...),
  **amplitude** ("with small/large amplitude"), and
  **speed** ("at slow/fast speed").
  e.g. "the camera tilts up with large amplitude at slow speed".

## Dialogue and voiceover

- Give speakers stable IDs `(S1)`, `(S2)`; establish who they are on first
  appearance. Dialogue goes in `<d>[Language] spoken text</d>` blocks with
  original punctuation.
- Voiceover must say "off-screen voiceover" and state that lips stay closed.

## Audio

The model generates the soundtrack from the prompt: name concrete diegetic
sounds inline with the action (in the description), keep ambience in
`overall_soundscape`, and score in `non_diegetic_music`.

## Reference-guided prompts (ref2va)

From the separate `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`. Ref2va prompts are
**one string with six ordered, labeled sections**:

```
subject_definitions: <define reusable labels>
summary: [<task types>] <one-paragraph gist of the target video>
retention_analysis: <per label: how faithfully it is kept>
detailed_description: <the [Shot N] narrative, as in the base guide>
overall_soundscape: <as in the base guide>
non_diegetic_music: <as in the base guide>
```

**Labels** — assigned in `subject_definitions`, then used consistently:
- `<Subject N>` — a reusable element (person/object/scene/style) *abstracted
  from* the references; can combine sources ("appearance from <Picture 1>,
  motion from <Video 1>"). This is the normal way to ground a person's
  identity from several photos.
- `<Picture N>` — the N-th reference image **in the order passed** (`--ref`
  order = Picture numbering). Standalone only when the image anchors a
  concrete frame/composition; otherwise cite it inside a subject definition.
- `<Video N>` / `<Audio N>` — whole-video relationships (continuation,
  editing, rhythm) / audio to copy or mimic. Audio tied to a speaker is
  written `<Audio 1> (S1)`.

**summary task types** (combine with ` + `): `keyframe completion`,
`reference generation`, `video editing`, `video continuation`,
`audio reuse`, `audio reference`.

**retention_analysis markers** — visual: `fully_preserved`,
`partially_preserved`, `attribute_transfer`, `weak_reference`;
audio: `fully_copy`, `partially_copy`, `reference`, `weak_reference`.
Format: `<Subject 1> (appears in [Shot 1]): fully_preserved - <what is kept>.`

**detailed_description** follows the base-guide shot rules (first shot
untimestamped, later shots `[Shot N] At MM:SS.mmm, ...`, camera motion as
type + amplitude + speed, dialogue in `<d>[Language] ...</d>` with stable
`(S1)` speaker IDs); reference labels appear at first mention and wherever
their role applies. When a referenced subject speaks: `<Subject 2> (S1) says,
<d>[English] ...</d>`.

## Example skeleton (FL2VA)

```
integrated_multimodal_description: Picture 1 aligns with the 0.00-second
mark; Picture 2 aligns with the 8.00-second mark. [Shot 1] Live-action,
cinematic. The cyclist begins in the position established by Picture 1,
holding a closed umbrella beside her bicycle. The camera pulls out with
small amplitude at slow speed as she raises the umbrella, and she settles
into the pose established by Picture 2. overall_soundscape: Light rain
patters on pavement; spokes click softly. non_diegetic_music: A sparse,
warm piano motif.
```

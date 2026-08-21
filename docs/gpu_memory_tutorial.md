# Fitting a 124 GB Model onto 80 GB Cards — a Tutorial

How to place, split, offload, and shrink large diffusion models with diffusers,
using this repo's real model as the running example. MiniMax-H3 is ~124 GB of
bf16 weights (61.7 GB DiT transformer + 62.1 GB Qwen3-VL conditioner, plus
video/audio VAEs) and our cards are 80 GB H100s — nothing fits anywhere without
one of these techniques. Two of them are live in `scripts/minimax_h3.py` today;
the rest are the upgrade path.

Code marked **[in this repo]** runs here today. Code marked **[pattern]** is a
worked sketch against diffusers main (our pinned rev) that you'd adapt — modular
pipelines are a moving API, so verify signatures against the pinned commit.

## 0. First, know which wall you're hitting

GPU memory during inference is spent on two very different things:

- **Weights** — fixed at load time. Solved by *placement* (which card holds
  what), *offloading* (weights hop on/off the GPU), or *quantization* (weights
  get smaller).
- **Activations & latents** — scale with your request: pixels × frames for the
  denoiser, reference count for the conditioner (each image ref becomes vision
  tokens at a fixed 2048-px short edge, so input resolution is irrelevant).
  Solved by *freeing weights to make room*, or by *shrinking the request*.

Diagnose before choosing a technique — an OOM traceback names the GPU:

```python
# After a run (or in an except block around the pipe call):
for d in range(torch.cuda.device_count()):
    print(f"cuda:{d} peak allocated: {torch.cuda.max_memory_allocated(d)/2**30:.1f} GiB")
```

Our two walls, measured on 2× H100:
- fl2va/ref2va OOM above ~**53M pixels × frames** → activation wall on the
  *denoise* card (weights leave only ~18 GB headroom).
- ref2va OOMs at **3 image refs** → activation wall on the *conditioner* card,
  during encoding.

## 1. Component-level placement — one whole model per card **[in this repo]**

The coarsest split: give each component its own GPU. No model is divided; you
just stop them from sharing. This is what `load_model()` does for `num_gpus >= 2`:

```python
# scripts/minimax_h3.py (two-GPU path), abridged:
blocks = ModularPipeline.from_pretrained(model_id).blocks.get_workflow(workflow)

# Carve the conditioning stage (media prep + Qwen3-VL) out of the block graph...
conditioner_blocks = SequentialPipelineBlocks.from_blocks_dict(
    {name: blocks.sub_blocks.pop(name)
     for name in ("before_encode", "text_encoder") if name in blocks.sub_blocks}
)

# ...and give each stage its own ComponentsManager pinned to its own card.
text_manager = ComponentsManager()
text_manager.enable_auto_cpu_offload(device="cuda:1")      # conditioner lives on cuda:1
conditioner = conditioner_blocks.init_pipeline(model_id, components_manager=text_manager)
conditioner.load_components(dtype=torch.bfloat16)

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda:0")           # transformer + VAEs on cuda:0
pipe = blocks.init_pipeline(model_id, components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
```

At generation time the two stages hand off through the pipeline `state`:

```python
state = conditioner(prompt=prompt, **media_kwargs)   # runs on cuda:1
results = pipe(state=state, num_frames=..., ...)     # runs on cuda:0
```

For a *standard* (non-modular) pipeline the same idea is one argument:

```python
# [pattern] — standard pipelines only; "balanced" is the only supported map
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                         device_map="balanced")
```

**Properties:** simple, no cross-card traffic during the denoise loop, and both
stages keep native speed. **Limit:** the biggest single component still has to
fit one card — 62 GB transformer on an 80 GB card leaves only ~18 GB for
activations. That's exactly our pixels×frames ceiling. When you hit it, you
need technique 3.

## 2. CPU offloading — trade speed for memory **[in this repo]**

When there's one GPU (or not enough), components live in host RAM and visit the
GPU only while needed. The single-GPU path in `load_model()`:

```python
# scripts/minimax_h3.py (one-GPU path):
manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(model_id, components_manager=manager)
pipe.load_components(workflow=workflow, dtype=torch.bfloat16)
manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
```

`enable_auto_cpu_offload` watches usage: when the text encoder is done, it's
evicted to make room for the transformer, and so on. Needs ~150–200 GB host RAM
and is much slower (every stage pays a PCIe transfer), but it makes 124 GB run
on one 80 GB card.

The standard-pipeline family of the same idea, from coarse to fine:

```python
# [pattern] Whole components hop on/off (good default, modest slowdown):
pipe.enable_model_cpu_offload()

# [pattern] Layer-by-layer (extreme memory savings, extreme slowdown):
pipe.enable_sequential_cpu_offload()

# [pattern] Block-level with prefetch on a CUDA stream — the middle ground.
# Groups of transformer blocks stream in ahead of use, hiding most transfer cost:
pipe.transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level", num_blocks_per_group=2,
    use_stream=True,
)
```

**Rule of thumb:** offloading rescues *weights* memory only. If your OOM is an
activation spike inside one forward pass (our 3-ref problem), offloading the
other components buys headroom, but the spike itself doesn't shrink.

## 3. Layer-level sharding — split one model across cards **[pattern]**

The technique that moves our ceiling: divide the 62 GB transformer itself,
~31 GB of layers per card, leaving ~45 GB of activation headroom everywhere.
diffusers/accelerate do this natively at load time with `device_map="auto"`:

```python
import torch
from diffusers import AutoModel

transformer = AutoModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3",
    subfolder="transformer",          # "transformer_ref" for ref2va
    torch_dtype=torch.bfloat16,
    device_map="auto",                # accelerate plans a layer split...
    max_memory={0: "38GiB", 1: "38GiB"},  # ...within these per-card budgets
)
# Then hand the sharded model to the pipeline in place of its own copy:
pipe.update_components(transformer=transformer)
```

Three things to understand before trusting it:

1. **The split is planned from free VRAM at load time.** Load the transformer
   while 62 GB of Qwen3-VL already sits on a card and the planner will cram
   layers onto the emptier card or spill to CPU. Either give explicit
   `max_memory` budgets (above) or sequence your loads (technique 4).
2. **Execution is sequential, not parallel.** accelerate places hooks that move
   the activation from card 0 to card 1 at the split point. Card 1 waits for
   card 0 every step. This is memory relief — steps get *slightly slower*, not
   faster.
3. **The model class must declare split points** (`_no_split_modules`), so
   attention blocks etc. are never divided internally. Big video transformers
   on diffusers main generally do; if it's missing, `device_map="auto"` errors.

## 4. Encode-then-free — sequence loads so everything fits **[pattern]**

The conditioner runs *once*, before the denoise loop. Nothing forces it to keep
its 62 GB resident afterwards. Orchestrating that by hand is the cleanest route
to sharding without fighting the load-time planner:

```python
# 1) Load ONLY the conditioner stage and encode (both cards are empty, so it
#    can even use device_map="auto" itself if refs are heavy):
conditioner = conditioner_blocks.init_pipeline(model_id, components_manager=text_manager)
conditioner.load_components(dtype=torch.bfloat16)
state = conditioner(prompt=prompt, **media_kwargs)      # embeddings now live in `state`

# 2) Release the conditioner entirely — embeddings are computed, weights are dead:
del conditioner
text_manager = None
gc.collect()
torch.cuda.empty_cache()                                # both cards now ~empty

# 3) NOW load the transformer sharded across both cards (technique 3) and denoise:
pipe.load_components(dtype=torch.bfloat16)              # with the sharded transformer
results = pipe(state=state, num_frames=..., ...)
```

The trade: you reload the conditioner on every process launch, so per-run
startup grows, and you can't cheaply re-prompt within one process. For batch
generation (seed sweeps of one prompt) that's irrelevant — encode once, denoise
many.

## 5. Sequential reference encoding — shrink the spike itself **[pattern]**

Our 3-ref OOM is a *single-forward-pass* spike: all references become
2048-px-short-edge vision tokens in one batched Qwen3-VL call. No placement or
offload trick shrinks one forward pass — but encoding refs one at a time does,
making peak memory ~one ref instead of the sum:

```python
# Conceptual — this loop lives inside the text-encoder block, so implementing
# it means overriding that block rather than calling the pipeline differently:
ref_embeddings = []
for ref in references:
    ref_embeddings.append(vision_tower(preprocess(ref)))   # peak = ONE ref
    torch.cuda.empty_cache()
embeddings = assemble(prompt_tokens, ref_embeddings)        # cat along sequence dim
```

The caveat is honesty about where the seam is: the batched call happens inside
the modular pipeline's `text_encoder` block, so this is the one technique that
means modifying pipeline internals (subclass the block, or encode refs through
the processor/vision tower directly and inject the embeddings). Verify against
the block source at our pinned diffusers rev before building it.

## 6. Small wins: VAE placement and attention backends **[pattern]**

- **Move the VAEs off the denoise card.** They only run before/after the
  denoise loop, so parking them on the conditioner's card (or CPU) returns a
  few GB to where the pressure is: `pipe.vae.to("cuda:1")` — or under a
  ComponentsManager, pass `device="cuda:1"` when loading those components.
- **Efficient attention kernels** cut *activation* memory as well as time.
  The script already tries Flash Attention 3 on H100
  (`transformer.set_attention_backend("_flash_3_hub")` — unavailable on our
  nodes, see `cluster_environment.md`); default SDPA is the fallback and fine.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — not a memory
  reduction, but it lets the allocator grow segments instead of fragmenting.
  Mandatory headroom at our sizes; already in the launch template.

## 7. Quantization — make the weights smaller **[pattern]**

The only technique that reduces weight memory outright. bf16 → 8-bit halves it;
4-bit quarters it:

```python
from diffusers import AutoModel, BitsAndBytesConfig

transformer = AutoModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3", subfolder="transformer",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.bfloat16,
)
```

(torchao fp8 is the H100-native alternative; same shape, different config
object.) We treat this as the last resort here: MiniMax-H3 is
guidance-distilled — CFG behavior is baked into the weights — so it's plausibly
more sensitive to quantization error than a CFG model, and techniques 3–5 reach
our goals without touching output quality. If you try it, A/B against bf16 on a
fixed seed first.

## 8. What diffusers does NOT give you

- **Tensor parallelism for inference** — both cards computing each layer
  together for a speedup. Layer sharding (technique 3) is sequential; if you
  need faster steps rather than more memory, that's custom engineering
  (or a different serving stack), not a flag.
- **Multi-node inference** — a second node's GPUs are invisible to a
  single-process diffusers run. On this cluster, 2× H100 per allocation is the
  working envelope.

## 9. Choosing: symptom → technique

| Symptom | Reach for | Cost |
|---|---|---|
| Model won't load, multiple GPUs available | 1. Component placement | none — do this first |
| Model won't load, one GPU | 2. CPU offload (auto/model/group) | speed |
| Denoise-card OOM at big canvas / many frames | 3. Layer sharding (+ 4 to load clean, + 6 VAE move) | slightly slower steps, more moving parts |
| Conditioner OOM from many image refs | 5. Sequential ref encoding | block-level surgery |
| Still short after all placement tricks | 7. Quantization | output quality risk |
| Steps too slow (not a memory problem) | nothing here — fewer steps, smaller canvas, or FA3 | — |

**This repo today:** technique 1 (2-GPU) and 2 (1-GPU), plus the small wins in
6 that apply. **The planned upgrade** for full-canvas ref2va with 3 refs:
4 (encode-then-free) + 3 (shard the transformer) + 6 (VAEs off the denoise
card) + 5 (sequential refs) — in that order, measuring peak memory (section 0)
after each step, because each one may already be enough.

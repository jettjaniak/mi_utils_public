from .ext_imports import *
from .prompts import BatchedPrompts


# Neel's original composition scores, we don't use them anymore
@typechecked
def compute_ov_comp_scores_by_layer_to(model: EasyTransformer, mode: str) -> Dict[int, torch.Tensor]:
    all_ov_comp_scores = model.all_composition_scores(mode)
    ret = {}
    for layer_to in range(1, model.cfg.n_layers):
        # heads_to in first dim
        ret[layer_to] = all_ov_comp_scores[:layer_to, :, layer_to, :].permute(2, 0, 1)
    return ret

# Using Neel's original composition scores, we don't use them anymore
@typechecked
def compute_ov_comp_scores_by_layer_to_by_mode(model: EasyTransformer) -> Dict[str, Dict[int, torch.Tensor]]:
    return {mode: compute_ov_comp_scores_by_layer_to(model, mode) for mode in "QKV"}

"""
@typechecked
def run_model_without_connection(
    model: EasyTransformer,
    prompt: TT["batch", "pos"],
    layerhead_from: Tuple[int, int],
    layerhead_to: Tuple[int, int],
    clean_cache: ActivationCache,
    corrupt_cache: ActivationCache
) -> TT["batch", "pos", "d_vocab"]:
    use_attn_result = model.cfg.use_attn_result
    model.cfg.use_attn_result = True
    layer_from, head_from = layerhead_from
    layer_to, head_to = layerhead_to

    def subtract_head_from_result(resid_pre, hook):
        resid_pre[:, :] -= clean_cache[f"blocks.{layer_from}.attn.hook_result"][
            :, :, head_from
        ]
        resid_pre[:, :] += corrupt_cache[f"blocks.{layer_from}.attn.hook_result"][
            :, :, head_from
        ]

    head_to_attn_result = None

    def store_head_to_result(layer_to_attn_result, hook):
        nonlocal head_to_attn_result
        head_to_attn_result = layer_to_attn_result[:, :, head_to]

    model.run_with_hooks(
        prompt,
        fwd_hooks=[
            (f"blocks.{layer_to}.hook_resid_pre", subtract_head_from_result),
            (f"blocks.{layer_to}.attn.hook_result", store_head_to_result),
        ],
    )

    # Now run with the intermediate one
    def insert_head_to_result(layer_to_attn_result, hook):
        layer_to_attn_result[:, :, head_to] = head_to_attn_result

    model.reset_hooks()
    logits = model.run_with_hooks(
        prompt,
        fwd_hooks=[(f"blocks.{layer_to}.attn.hook_result", insert_head_to_result)],
    )
    model.cfg.use_attn_result = use_attn_result
    return logits
"""

@typechecked
def run_model_without_connection(
    model: EasyTransformer,
    prompt: TT["batch", "pos"],
    layerhead_from: Tuple[int, int],
    layerhead_to: Tuple[int, int],
    clean_cache: Optional[ActivationCache],
    corrupt_cache: Optional[ActivationCache],
    modes: str = "qkv"
) -> TT["batch", "pos", "d_vocab"]:
    use_attn_result = model.cfg.use_attn_result
    model.cfg.use_attn_result = True
    layer_from, head_from = layerhead_from
    layer_to, head_to = layerhead_to
    # Note that this only patches the direct connection between two heads.
    # Inserting q+k+v is equivalent to inserting hook_result iff nothing is going via other heads
    # in any layer > later_from and <= layer_to.
    # E.g. if L0H0 affects L1H7 but we're only inserting L1H0's qkv then we're missing L1H7's changed k.
    def subtract_head_from_result(resid_pre, hook):
        resid_pre[:, :] -= clean_cache[f"blocks.{layer_from}.attn.hook_result"][
            :, :, head_from
        ]
        if corrupt_cache is not None:
            resid_pre[:, :] += corrupt_cache[f"blocks.{layer_from}.attn.hook_result"][
                :, :, head_from
            ]

    head_to_qkv = {}

    def store_head_to_qkv(layer_to_attn_qkv, hook):
        mode = hook.name[-1]
        assert mode in modes
        # batch, pos, head, dim
        head_to_qkv[mode] = layer_to_attn_qkv[:, :, head_to]

    store_qkv_hooks = [(f"blocks.{layer_to}.attn.hook_{mode}", store_head_to_qkv) for mode in modes]
    model.run_with_hooks(
        prompt,
        fwd_hooks=[
            (f"blocks.{layer_to}.hook_resid_pre", subtract_head_from_result),
        ] + store_qkv_hooks,
    )

    # Now run with the intermediate one
    def insert_head_to_qkv(layer_to_attn_qkv, hook):
        mode = hook.name[-1]
        assert mode in modes
        layer_to_attn_qkv[:, :, head_to] = head_to_qkv[mode]

    insert_qkv_hooks = [(f"blocks.{layer_to}.attn.hook_{mode}", insert_head_to_qkv) for mode in modes]
    model.reset_hooks()
    logits = model.run_with_hooks(
        prompt,
        fwd_hooks=insert_qkv_hooks,
    )
    model.cfg.use_attn_result = use_attn_result
    return logits


@typechecked
def compute_patching_comp_scores(
    *,
    model: EasyTransformer,
    batched_prompts: BatchedPrompts,
    layer_to: int,
    clean_cache: ActivationCache,
    corrupt_cache: Optional[ActivationCache],
    clean_logit_diff: TT["batch"],
    modes: str = "qkv",
    relative = False,
) -> TT["n_heads", "layers_from", "n_heads"]:
    # batch, heads_to, layers_from, heads_from
    batch_size = batched_prompts.clean_tokens.shape[0]
    assert clean_logit_diff.shape[0] == batch_size
    scores = torch.zeros(
        batch_size, model.cfg.n_heads, layer_to, model.cfg.n_heads
    ).cuda()
    for head_to in range(model.cfg.n_heads):
        for layer_from in range(layer_to):
            for head_from in range(model.cfg.n_heads):
                patched_logits = run_model_without_connection(
                    model,
                    batched_prompts.clean_tokens,
                    layerhead_from=(layer_from, head_from),
                    layerhead_to=(layer_to, head_to),
                    clean_cache=clean_cache,
                    corrupt_cache=corrupt_cache,
                    modes=modes
                )
                patched_logit_diff = batched_prompts.logit_diff(patched_logits)
                if relative:
                    scores[:, head_to, layer_from, head_from] = (
                        patched_logit_diff - clean_logit_diff
                        ) / clean_logit_diff.mean(dim=0)
                else:
                    scores[:, head_to, layer_from, head_from] = (
                        patched_logit_diff - clean_logit_diff
                        )
    return scores.mean(dim=0)

@typechecked
def compute_patching_comp_scores_by_layer_to(
    *,
    model: EasyTransformer,
    batched_prompts: BatchedPrompts,
    clean_cache: ActivationCache,
    corrupt_cache: ActivationCache,
    clean_logit_diff: TT["batch"],
    modes: str = "qkv"
) -> Dict[int, torch.Tensor]:
    ret = {}
    for layer_to in range(1, model.cfg.n_layers):
        ret[layer_to] = compute_patching_comp_scores(
            model=model,
            batched_prompts=batched_prompts,
            layer_to=layer_to,
            clean_cache=clean_cache,
            corrupt_cache=corrupt_cache,
            clean_logit_diff=clean_logit_diff,
            modes=modes
        )
    return ret
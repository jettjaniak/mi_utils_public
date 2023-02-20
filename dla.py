from .misc import *

@typechecked
def residual_stack_to_logit_diff(
    residual_stack: TT["components", "batch", "d_model"],
    cache: ActivationCache,
    logit_diff_directions: TT["batch", "d_model"],
) -> TT["components"]:
    _, batch_size, _ = residual_stack.shape
    # pos_slice=-1, because we only care about last token prediction
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, pos_slice=-1)
    logit_diff = einsum(
        "components batch d_model, batch d_model -> components",
        scaled_residual_stack,
        logit_diff_directions,
    ) / batch_size
    return logit_diff


@typechecked
def per_head_logit_diff(
    cache: ActivationCache, logit_diff_directions: TT["batch", "d_model"]
):
    per_head_residual = cache.stack_head_results(layer=-1, pos_slice=-1)
    per_head_logit_diffs = residual_stack_to_logit_diff(
        per_head_residual, cache, logit_diff_directions
    )
    per_head_logit_diffs = einops.rearrange(
        per_head_logit_diffs,
        "(layer head_index) -> layer head_index",
        layer=cache.model.cfg.n_layers,
        head_index=cache.model.cfg.n_heads,
    )
    return per_head_logit_diffs


@typechecked
def get_logit_diff_directions(
    model: EasyTransformer,
    correct_tokens: TT["batch", "correct_tokens", torch.int64],
    wrong_tokens: TT["batch", "wrong_tokens", torch.int64],
) -> TT["batch", "d_model"]:
    correct_direction = model.tokens_to_residual_directions(correct_tokens).mean(dim=1)
    wrong_direction = model.tokens_to_residual_directions(wrong_tokens).mean(dim=1)
    return correct_direction - wrong_direction

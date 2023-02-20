from .misc import *
device="cuda"
def test_prompt_logit_swap(model, prompt, a, b, pos=-1):
    model.reset_hooks()
    prompt_main = prompt.format(a=a, b=b)
    prompt_swap = prompt.format(aet_utils=b, b=a)
    idx_a = torch.tensor(model.tokenizer.encode(a)).item()
    idx_b = torch.tensor(model.tokenizer.encode(b)).item()
    logits_main = model(prompt_main, prepend_bos=False)
    logits_swap = model(prompt_swap, prepend_bos=False)

    correctness_main = logits_main[:,pos,idx_a] - logits_main[:,pos,idx_b]
    correctness_swap = logits_swap[:,pos,idx_b] - logits_swap[:,pos,idx_a]
    return correctness_main.item(), correctness_swap.item()


def patch(activations, hook, cache=None, pos=None, head=None):
    """General patching function to patch anything.
        cache: cache values to patch-in. Ablate if None
        pos: token-position (optional)
        head: head (optional)
    """
    pos = slice(0, None) if pos is None else pos
    head = slice(0, None) if head is None else head

    if len(activations.shape) == 4:
        # Dimensions for attn.hook_result: Batch, Pos, Head, d_embed
        fill_value = 0 if cache is None else cache[hook.name][:, pos, head, :]
        activations[:, pos, head, :] = fill_value
    elif len(activations.shape) == 3:
        # Dimensions for hook_resid: Batch, Pos, d_embed
        fill_value = 0 if cache is None else cache[hook.name][:, pos, :]
        activations[:, pos, :] = fill_value
    else:
       raise ValueError(f"Unknown activation shape {activations.shape}")
    
    return activations

@typechecked
def correct_logprob(logits: TT["batch", "pos", "d_vocab"], correct_tokens: TT["batch", "n_correct"], pos: int = -1) -> TT["batch"]:
    pos_logprobs = torch.log_softmax(logits[:, pos, :], dim=-1)
    max_correct, _ = torch.gather(pos_logprobs, index=correct_tokens, dim=1).max(dim=1)
    return max_correct

def patch_stream(activations: TT["batch", "pos", "d_embed"], hook, cache=None, pos=None):
    """General patching function to patch anything.
        cache: cache values to patch-in. Ablate if None
        pos: token-position (optional)
    """
    pos = slice(0, None) if pos is None else pos
    fill_value = 0 if cache is None else cache[hook.name][:, pos, :]
    activations[:, pos, :] = fill_value
    return activations

def patch_head(activations: TT["batch", "pos", "head", "d_embed"], hook, cache=None, pos=None, head=None):
    """General patching function to patch anything.
        cache: cache values to patch-in. Ablate if None
        pos: token-position (optional)
    """
    pos = slice(0, None) if pos is None else pos
    head = slice(0, None) if head is None else head
    fill_value = 0 if cache is None else cache[hook.name][:, pos, head, :]
    activations[:, pos, head, :] = fill_value
    return activations

def residual_stream_patching(model, tokens_to: TT["batch", "pos", "d_vocab"], cache_from,
                             correct_tokens: TT["batch", "n_correct"],
                             wrong_tokens:  TT["batch", "n_wrong"] = None,
                             logprob=False, topk=0):
    """Residual stream patching
        Patch-in values from cache_from into a run with tokens_to
    """
    if wrong_tokens is None and not logprob:
       raise ValueError("wrong_tokens cannot be None if logprob is False")
    # Clean run to compare to
    model.reset_hooks()
    batch_size, n_ctx = tokens_to.shape
    n_layers = model.cfg.n_layers
    logits_to_unpatched = model(tokens_to)
    if topk:
        # Batch index 0 only for topk guesses
        final_logits_to_unpatched = logits_to_unpatched[0, -1]
        topk_indices_patched = torch.zeros(n_layers+1, n_ctx, topk, device=device, dtype=torch.int)
        topk_logits_patched = torch.zeros(n_layers+1, n_ctx, topk, device=device, dtype=torch.float)
        topk_guesses_patched = [[["" for _ in range(topk)] for _ in range(n_ctx)] for _ in range(n_layers+1)]
        topk_logits_unpatched, topk_indices_unpatched = torch.topk(final_logits_to_unpatched, k=topk, dim=-1)
        topk_guesses_unpatched = [f"{i}: |{model.tokenizer.decode(topk_indices_unpatched[i])}| ({topk_logits_unpatched[i]:.2f})" for i in range(topk)]
    if logprob:
        # pos is always -1 because that's the "answer", i.e. what correct_tokens refers to
        metric = lambda logits_to_unpatched, correct_tokens, wrong_tokens: correct_logprob(logits_to_unpatched, correct_tokens, pos=-1)
    else:
        metric = lambda logits_to_unpatched, correct_tokens, wrong_tokens: logit_diff(logits_to_unpatched, correct_tokens, wrong_tokens, pos=-1)
    unpatched_metric = metric(logits_to_unpatched, correct_tokens, wrong_tokens)
    # Patched runs
    patched_residual_stream_diff = torch.zeros(n_layers+1, n_ctx, device=device, dtype=torch.float32)
    for layer in range(n_layers+1):
        layer_name = f"blocks.{layer}.hook_resid_pre" if layer < n_layers else f"blocks.{layer-1}.hook_resid_post"
        for position in range(n_ctx):
            hook_fn = partial(patch_stream, pos=position, cache=cache_from)
            logits_patched = model.run_with_hooks(tokens_to, fwd_hooks = [(layer_name, hook_fn)])
            if topk:
                topk_logits_patched[layer, position, :], topk_indices_patched[layer, position, :] = torch.topk(logits_patched[0, -1], k=topk, dim=-1)
                topk_guesses_patched[layer][position] = [f"{i}: |{model.tokenizer.decode(topk_indices_patched[layer, position, i])}| ({topk_logits_patched[layer, position, i]:.2f})" for i in range(topk)]

            patched_residual_stream_diff[layer, position] = (metric(logits_patched, correct_tokens, wrong_tokens) - unpatched_metric).mean()
    if topk:
        return patched_residual_stream_diff, topk_guesses_unpatched, topk_guesses_patched
    return patched_residual_stream_diff

def all_head_patching(model, tokens_to: TT["batch", "pos", "d_vocab"], cache_from,
                             correct_tokens: TT["batch", "n_correct"],
                             wrong_tokens:  TT["batch", "n_wrong"] = None,
                             logprob=False, pos=None, modes=None, silent=False):
    """Residual stream patching
        Patch-in values from cache_from into a run with tokens_to
    """
    if modes is not None:
        if not silent and pos is not None:
            print(f"Warning: Patching {modes} at pos {pos}")
    if wrong_tokens is None and not logprob:
       raise ValueError("wrong_tokens cannot be None if logprob is False")
    # Clean run to compare to
    model.reset_hooks()
    batch_size, n_ctx = tokens_to.shape
    logits_to_unpatched = model(tokens_to)
    if logprob:
        # pos is always -1 because that's the "answer", i.e. what correct_tokens refers to
        metric = lambda logits_to_unpatched, correct_tokens, wrong_tokens: correct_logprob(logits_to_unpatched, correct_tokens, pos=-1)
    else:
        metric = lambda logits_to_unpatched, correct_tokens, wrong_tokens: logit_diff(logits_to_unpatched, correct_tokens, wrong_tokens, pos=-1)
    unpatched_metric = metric(logits_to_unpatched, correct_tokens, wrong_tokens)
    # Patched runs
    patched_residual_stream_diff = torch.zeros(model.cfg.n_layers, device=device, dtype=torch.float32)
    for layer in range(model.cfg.n_layers):
        head = None
        if modes is None:
            layer_name = f"blocks.{layer}.attn.hook_result"
            hook_fn = partial(patch_head, head=head, cache=cache_from, pos=pos)
            fwd_hooks = [(layer_name, hook_fn)]
        else:
            fwd_hooks = []
            for mode in modes:
                assert mode in "qkv", "Use lowercase modes"
                layer_name = f"blocks.{layer}.attn.hook_{mode}"
                hook_fn = partial(patch_head, head=head, cache=cache_from, pos=pos)
                fwd_hooks.append((layer_name, hook_fn))
        logits_patched = model.run_with_hooks(tokens_to, fwd_hooks=fwd_hooks)
        patched_residual_stream_diff[layer] = (metric(logits_patched, correct_tokens, wrong_tokens) - unpatched_metric).mean()
    return patched_residual_stream_diff

def residual_stream_ablating(model, tokens_to: TT["batch", "pos", "d_vocab"],
                             correct_tokens: TT["batch", "n_correct"],
                             wrong_tokens:  TT["batch", "n_wrong"] = None,
                             logprob=False):
    return residual_stream_patching(model=model, tokens_to=tokens_to, cache_from=None,
                             correct_tokens=correct_tokens, wrong_tokens=wrong_tokens,
                             logprob=logprob)


def head_result_patching(model, tokens_to: TT["batch", "pos", "d_vocab"], cache_from,
                             correct_tokens: TT["batch", "n_correct"],
                             wrong_tokens:  TT["batch", "n_wrong"] = None,
                             logprob=False, pos=None, modes=None, silent=False):
    """Residual stream patching
        Patch-in values from cache_from into a run with tokens_to
    """
    if modes is not None:
        if not silent and pos is not None:
            print(f"Warning: Patching {modes} at pos {pos}")

    if wrong_tokens is None and not logprob:
       raise ValueError("wrong_tokens cannot be None if logprob is False")
    # Clean run to compare to
    model.reset_hooks()
    batch_size, n_ctx = tokens_to.shape
    logits_to_unpatched = model(tokens_to)
    if logprob:
        # pos is always -1 because that's the "answer", i.e. what correct_tokens refers to
        metric = lambda logits_to_unpatched, correct_tokens, wrong_tokens: correct_logprob(logits_to_unpatched, correct_tokens, pos=-1)
    else:
        metric = lambda logits_to_unpatched, correct_tokens, wrong_tokens: logit_diff(logits_to_unpatched, correct_tokens, wrong_tokens, pos=-1)
    unpatched_metric = metric(logits_to_unpatched, correct_tokens, wrong_tokens)
    # Patched runs
    patched_residual_stream_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            if modes is None:
                layer_name = f"blocks.{layer}.attn.hook_result"
                hook_fn = partial(patch_head, head=head, cache=cache_from, pos=pos)
                fwd_hooks = [(layer_name, hook_fn)]
            else:
                fwd_hooks = []
                for mode in modes:
                    assert mode in "qkv", "Use lowercase modes"
                    layer_name = f"blocks.{layer}.attn.hook_{mode}"
                    hook_fn = partial(patch_head, head=head, cache=cache_from, pos=pos)
                    fwd_hooks.append((layer_name, hook_fn))
            logits_patched = model.run_with_hooks(tokens_to, fwd_hooks=fwd_hooks)
            patched_residual_stream_diff[layer, head] = (metric(logits_patched, correct_tokens, wrong_tokens) - unpatched_metric).mean()
    return patched_residual_stream_diff


def indiv_head_result_patching(model, tokens_to: TT["batch", "pos", "d_vocab"],
                               tokens_from: TT["batch", "pos", "d_vocab"],
                               correct_tokens: TT["batch", "n_correct"],
                               wrong_tokens:  TT["batch", "n_wrong"],
                             logprob=False, pos=None, head=None, layer=False):
    """Residual stream patching
        Patch-in values from tokens_from into a run with tokens_to
    """
    batch_size, n_ctx = tokens_to.shape
    logits_to_unpatched = model(tokens_to)
    metric = lambda logits_to_unpatched, correct_tokens, wrong_tokens: logit_diff(logits_to_unpatched, correct_tokens, wrong_tokens, pos=-1)
    unpatched_metric = metric(logits_to_unpatched, correct_tokens, wrong_tokens)
    # Patched runs
    cache_from = model.run_with_cache(tokens_from)[1]
    layer_name = f"blocks.{layer}.attn.hook_result"
    hook_fn = partial(patch_head, head=head, cache=cache_from, pos=pos)
    fwd_hooks = [(layer_name, hook_fn)]
    logits_patched = model.run_with_hooks(tokens_to, fwd_hooks=fwd_hooks)
    return (metric(logits_patched, correct_tokens, wrong_tokens) - unpatched_metric).mean()


def head_result_ablating(model, tokens_to: TT["batch", "pos", "d_vocab"],
                             correct_tokens: TT["batch", "n_correct"],
                             wrong_tokens:  TT["batch", "n_wrong"] = None,
                             logprob=False, pos=None):
    return head_result_patching(model=model, tokens_to=tokens_to, cache_from=None,
                             correct_tokens=correct_tokens, wrong_tokens=wrong_tokens,
                             logprob=logprob, pos=pos)




def average_correct_logprob(logits, indices, pos=-1, batch=None):
  # Batch, Pos, d_vocab
  final_logits = logits[:, pos, :]
  # Calculate logprobs
  final_logprobs = torch.log_softmax(final_logits, dim=-1).to("cpu")
  # Logprob of the answer
  # Todo: Figure out a way to make gather work for non-batches
  # answer_logprobs = final_logprobs.gather(dim=-1, index=indices)
  answer_logprobs = final_logprobs[0, indices]
  
  if batch is None:
    return answer_logprobs.mean()
  elif batch == "all":
    return answer_logprobs
  else:
    return answer_logprobs[batch]





@typechecked
def run_overwrite_heads(
    model: EasyTransformer,
    prompt: TT["batch", "pos"],
    # hook, dst, head, embed
    overwrites: List[Tuple[str, int, int, TT["batch", "src"]]],
    run_with_cache = True
    ):

    fwd_hooks = []

    def overwrite(activations, hook, dst=None, head=None, embed=None):
#        print("Overwriting", hook.name)
#        activations[:, pos, head, :] = embed
        activations[:, head, dst, :] = embed
        return activations

    for  hook, dst, head, embed in overwrites:
        fwd_hooks.append((hook,
            partial(overwrite, dst=dst, head=head, embed=embed)))

    # Store cache if requested
    if run_with_cache:
        output_cache = {}
        def store_in_cache(activations, hook):
            output_cache[hook.name] = activations
        for layer in range(model.cfg.n_layers):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_result", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_attn_scores", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_pattern", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_q", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_k", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_v", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.hook_resid_post", store_in_cache))
    
        logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
        return logits, output_cache
    else:
        logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
        return logits




@typechecked
def run_overwrite_pos_embed(
    model: EasyTransformer,
    prompt: TT["batch", "pos"],
    pairs_pos_embed: List[Tuple[int, TT["batch", "d_embed"]]],
    layer_pos_embed: int,
    run_with_cache = True
    ):

    fwd_hooks = []

    # Extract current positional embeddings to subract it later
    temp_cache = {}
    def store_pos_embed(pos_embed, hook):
        temp_cache[hook.name] = pos_embed
    fwd_hooks.append((f"hook_pos_embed", store_pos_embed))

    # Replace positional embeddings with the supplied ones
    def replace_pos_embed(activations, hook):
        for pos, embed in pairs_pos_embed:
            activations[:, pos, :] -= temp_cache["hook_pos_embed"][:, pos, :]
            activations[:, pos, :] += embed
        return activations
    fwd_hooks.append((f"blocks.{layer_pos_embed}.hook_resid_pre", replace_pos_embed))

    # Store cache if requested
    if run_with_cache:
        output_cache = {}
        def store_in_cache(activations, hook):
            output_cache[hook.name] = activations
        for layer in range(model.cfg.n_layers):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_result", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_attn_scores", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_pattern", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_q", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_k", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_v", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.hook_resid_post", store_in_cache))
    
        logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
        return logits, output_cache
    else:
        logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
        return logits




@typechecked
def run_overwrite_pos_embed_and_v(
    model: EasyTransformer,
    prompt: TT["batch", "pos"],
    pairs_pos_embed: List[Tuple[int, TT["batch", "d_embed"]]],
    layer_pos_embed: int,
    pairs_v: List[Tuple[int, TT["batch", "n_heads", "d_head"]]],
    layer_v: int,
    run_with_cache = True
    ):

    fwd_hooks = []

    # Extract current positional embeddings to subract it later
    temp_cache = {}
    def store_pos_embed(pos_embed, hook):
        temp_cache[hook.name] = pos_embed
    fwd_hooks.append((f"hook_pos_embed", store_pos_embed))

    # Replace positional embeddings with the supplied ones
    def replace_pos_embed(activations, hook):
        for pos, embed in pairs_pos_embed:
            activations[:, pos, :] -= temp_cache["hook_pos_embed"][:, pos, :]
            activations[:, pos, :] += embed
        return activations
    fwd_hooks.append((f"blocks.{layer_pos_embed}.hook_resid_pre", replace_pos_embed))

    # Replace v vectors with the supplied ones
    def replace_v(activations, hook):
        for pos, v in pairs_v:
            activations[:, pos, :, :] = v
        return activations
    fwd_hooks.append((f"blocks.{layer_v}.attn.hook_v", replace_v))

    # Store cache if requested
    if run_with_cache:
        output_cache = {}
        def store_in_cache(activations, hook):
            output_cache[hook.name] = activations
        for layer in range(model.cfg.n_layers):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_result", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_attn_scores", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_pattern", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_q", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_k", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_v", store_in_cache))
            fwd_hooks.append((f"blocks.{layer}.hook_resid_post", store_in_cache))
    
        logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
        return logits, output_cache
    else:
        logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
        return logits



#def old_residual_stream_patching(model, tokens_to, cache_from, answer_idx, title="Title"):
#    """Residual stream patching
#        Patch-in values from cache_from into a run with tokens_to
#    """
#    if cache_from is None:
#       print("INFO: cache_from is None, ablating instead")
#    if len(torch.tensor(tokens_to).shape) == 1:
#       print("INFO: Adding batch dimension to tokens_to")
#       tokens_to = torch.tensor([tokens_to])
#    if type(answer_idx) is int:
#       print("INFO: Adding batch dimension to answer_idx")
#       answer_idx = torch.tensor([answer_idx])
#    # Clean run to compare to
#    model.reset_hooks()
#    logits_clean = model(tokens_to)
#    average_correct_logprob_clean = average_correct_logprob(logits_clean, answer_idx)
#    # Patched runs
#    patched_residual_stream_diff = torch.zeros(model.cfg.n_layers+1, tokens_to.shape[1], device=device, dtype=torch.float32)
#    for layer in range(model.cfg.n_layers+1):
#        layer_name = et_utils.act_name("resid_pre", layer) if layer < model.cfg.n_layers else et_utils.act_name("resid_post", layer-1)
#        for position in range(tokens_to.shape[1]):
#            hook_fn = partial(patch, pos=position, cache=cache_from)
#
#            patched_logits = model.run_with_hooks(tokens_to, fwd_hooks = [(layer_name, hook_fn)])
#            patched_residual_stream_diff[layer, position] = average_correct_logprob(patched_logits, answer_idx) - average_correct_logprob_clean
#    # Plot
#    batch_idx_labels = 0
#    prompt_position_labels = [f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(tokens_to[batch_idx_labels]))]
#    imshow(patched_residual_stream_diff, xticks=prompt_position_labels, yticks=[*["resid_pre_"+str(n) for n in range(model.cfg.n_layers)], "resid_post_"+str(model.cfg.n_layers-1)], title=title, xlabel="Position", ylabel="Layer")
#
#def old_attn_result_patching_by_head(model, tokens_to, cache_from, answer_idx, pos=-1, title="Title"):
#    """Residual stream patching
#        Patch-in values from cache_from into a run with tokens_to
#    """
#    if cache_from is None:
#       print("INFO: cache_from is None, ablating instead")
#    if len(torch.tensor(tokens_to).shape) == 1:
#       print("INFO: Adding batch dimension to tokens_to")
#       tokens_to = torch.tensor([tokens_to])
#    if len(answer_idx.shape) == 1:
#       print("INFO: Adding batch dimension to answer_idx")
#       answer_idx = torch.tensor([answer_idx])
#    # Clean run to compare to
#    model.reset_hooks()
#    logits_clean = model(tokens_to)
#    average_correct_logprob_clean = average_correct_logprob(logits_clean, answer_idx)
#    # Patched runs
#    patched_residual_stream_diff = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)
#    for layer in range(model.cfg.n_layers):
#        layer_name = f"blocks.{layer}.attn.hook_result"
#        for head in range(model.cfg.n_heads):
#            hook_fn = partial(patch, pos=pos, head=head, cache=cache_from)
#
#            patched_logits = model.run_with_hooks(tokens_to, fwd_hooks = [(layer_name, hook_fn)])
#            patched_residual_stream_diff[layer, head] = average_correct_logprob(patched_logits, answer_idx) - average_correct_logprob_clean
#    # Plot
#    batch_idx_labels = 0
#    prompt_position_labels = [f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(tokens_to[batch_idx_labels]))]
#    imshow(patched_residual_stream_diff, title=title, xlabel="Head", ylabel="Layer")








@typechecked
def run_with_cache_and_swapped_pos_embed(
    model: EasyTransformer,
    prompt: TT["batch", "pos"],
    pos_swap_pairs: List[Tuple[int, int]],
    layer_to: int,
    ):
    print("This function is from Tuesday evening and deprecated, make a proper one plz")
    cache = {}
    def store_in_cache(pos_embed, hook):
        cache[hook.name] = pos_embed

    # Is it detecting newlines via prev token head stuff?
    def swap_pos_embed(resid_pre, hook):
        for p1, p2 in pos_swap_pairs:
            resid_pre[:, p1, :] -= cache[f"hook_pos_embed"][:, p1, :] 
            resid_pre[:, p1, :] += cache[f"hook_pos_embed"][:, p2, :] 
            resid_pre[:, p2, :] -= cache[f"hook_pos_embed"][:, p2, :] 
            resid_pre[:, p2, :] += cache[f"hook_pos_embed"][:, p1, :] 
        return resid_pre
    
    fwd_hooks=[
        (f"hook_pos_embed", store_in_cache),
        (f"blocks.{layer_to}.hook_resid_pre", swap_pos_embed),
    ]

    for layer in range(model.cfg.n_layers):
        fwd_hooks.append((f"blocks.{layer}.attn.hook_result", store_in_cache))
        fwd_hooks.append((f"blocks.{layer}.attn.hook_attn_scores", store_in_cache))
        fwd_hooks.append((f"blocks.{layer}.attn.hook_pattern", store_in_cache))

    logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)

    return logits, cache





@typechecked
def run_with_cache_and_swapped_pos_embed_in_mode_only(
    model: EasyTransformer,
    prompt: TT["batch", "pos"],
    pos_swap_pairs: List[Tuple[int, int]],
    clean_cache,
    layer_to=0,
    mode="v",
    ):
    print("This function is from Tuesday evening and deprecated, make a proper one plz")
    
    # Make corrupt run
    corrupt_cache = {}

    def store_in_corrupt_cache(act, hook):
        corrupt_cache[hook.name] = act.clone()

    def swap_pos_embed(activation, hook):
        for p1, p2 in pos_swap_pairs:
            activation[:, p1, :] -= clean_cache[f"hook_pos_embed"][:, p1, :] 
            activation[:, p1, :] += clean_cache[f"hook_pos_embed"][:, p2, :] 
            activation[:, p2, :] -= clean_cache[f"hook_pos_embed"][:, p2, :] 
            activation[:, p2, :] += clean_cache[f"hook_pos_embed"][:, p1, :] 
        return activation
    
    corrupted_hooks = [
        (f"blocks.{layer_to}.hook_resid_pre", swap_pos_embed),
        (f"blocks.{layer_to}.attn.hook_{mode}", store_in_corrupt_cache),
    ]

    _ = model.run_with_hooks(prompt, fwd_hooks=corrupted_hooks)

    # Make final run
    output_cache = {}
    def store_in_cache(pos_embed, hook):
        output_cache[hook.name] = pos_embed.clone()
    def insert_corrupted_mode(activation, hook):
        activation[:, :, :] = corrupt_cache[hook.name]
        return activation
    
    fwd_hooks=[(f"blocks.{layer_to}.attn.hook_{mode}", insert_corrupted_mode),
        (f"blocks.1.hook_resid_pre", swap_pos_embed),]
    for layer in range(model.cfg.n_layers):
        fwd_hooks.append((f"blocks.{layer}.attn.hook_result", store_in_cache))
        fwd_hooks.append((f"blocks.{layer}.attn.hook_attn_scores", store_in_cache))
        fwd_hooks.append((f"blocks.{layer}.attn.hook_pattern", store_in_cache))

    
    logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)

    return logits, output_cache





@typechecked
def run_with_cache_and_swapped_pos_embed_for_real(
    model: EasyTransformer,
    prompt: TT["batch", "pos"],
    pos_swap_pairs: List[Tuple[int, int]],
    swap_pairs_heads,
    heads_to_swap_attn,
    cache,
    freeze_head,
    ):
    output_cache = {}
    def store_in_cache(pos_embed, hook):
        output_cache[hook.name] = pos_embed

    def freeze(activation, hook):
        activation[...] = cache[hook.name]
        return activation

    def freeze2(activation, hook):
        activation[:, freeze_head, :, :] = cache[hook.name][:, freeze_head, :, :]
        return activation

    # Is it detecting newlines via prev token head stuff?
    def swap_pos_embed(activations, hook):
        activations = activations.clone()
        for p1, p2 in pos_swap_pairs:
            #hook_pos_embed[:, p1, :] -= cache[f"hook_pos_embed"][:, p1, :] 
            activations[:, p1, :] = cache[f"hook_pos_embed"][:, p2, :] 
            #hook_pos_embed[:, p2, :] -= cache[f"hook_pos_embed"][:, p2, :] 
            activations[:, p2, :] = cache[f"hook_pos_embed"][:, p1, :] 
        return activations
    
    def swap_attn(activations, hook):
        activations = activations.clone()
        for p1, p2 in swap_pairs_heads:
            activations[:, heads_to_swap_attn, p1, :] = cache[hook.name][:, heads_to_swap_attn, p2, :] 
            activations[:, heads_to_swap_attn, p2, :] = cache[hook.name][:, heads_to_swap_attn, p1, :] 
        return activations

    fwd_hooks=[
        (f"hook_pos_embed", swap_pos_embed),
        (f"blocks.0.attn.hook_pattern", swap_attn),
        #(f"blocks.0.attn.hook_pattern", freeze),
        (f"blocks.0.attn.hook_pattern", freeze2),
    ]

    for layer in range(model.cfg.n_layers):
        fwd_hooks.append((f"blocks.{layer}.attn.hook_result", store_in_cache))
        fwd_hooks.append((f"blocks.{layer}.attn.hook_attn_scores", store_in_cache))
        fwd_hooks.append((f"blocks.{layer}.attn.hook_pattern", store_in_cache))
        fwd_hooks.append((f"blocks.{layer}.ln1.hook_scale", freeze))

    logits = model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)

    return logits, output_cache

@typechecked
def get_args_logits_probs_tokens(
    model: HookedTransformer,
    tokens: TT["batch", "pos"],
    n_args: int = 10,
    offset: int = 7
) -> Tuple[TT["batch", "args"], TT["batch", "args"], TT["batch", "args"]]:
    
    logits = model(tokens)[:, -1]
    probs = torch.softmax(logits, dim=-1)

    batch_size = tokens.shape[0]
    args_indices = torch.arange(0, n_args) * 2 + offset
    args_indices = args_indices.unsqueeze(0).expand(batch_size, -1).cuda()
    pos_tokens = torch.gather(
        tokens,
        dim=-1,
        index=args_indices
    ).squeeze()

    pos_probs = torch.gather(
        probs,
        dim=-1,
        index=pos_tokens
    )

    pos_logits = torch.gather(
        logits,
        dim=-1,
        index=pos_tokens
    )

    return pos_logits, pos_probs, pos_tokens
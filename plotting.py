from .ext_imports import *


def imshow(
    tensor,
    xlabel="X",
    ylabel="Y",
    zlabel=None,
    xticks=None,
    yticks=None,
    c_midpoint=0.0,
    c_scale="RdBu",
    show=True,
    **kwargs
):
    tensor = et_utils.to_numpy(tensor)
    if "animation_frame" not in kwargs:
        assert len(tensor.shape) == 2
    else:
        assert len(tensor.shape) == 3
    xticks = xticks or range(tensor.shape[-1])
    yticks = yticks or range(tensor.shape[-2])
    xticks = [str(x) for x in xticks]
    yticks = [str(y) for y in yticks]
    if len(xticks) != len(set(xticks)):
        xticks = [f"{i}_{x}" for i, x in enumerate(xticks)]
    if len(yticks) != len(set(yticks)):
        yticks = [f"{i}_{y}" for i, y in enumerate(yticks)]
    labels = {"x": xlabel, "y": ylabel}
    if zlabel is not None:
        labels["color"] = zlabel
    fig = px.imshow(
        et_utils.to_numpy(tensor),
        x=xticks,
        y=yticks,
        labels=labels,
        color_continuous_midpoint=c_midpoint,
        color_continuous_scale=c_scale,
        **kwargs
    )
    if show:
        fig.show()
    else:
        return fig


def imshow2(tensor, renderer=None, **kwargs):
    # Deprecated, don't use (Neel's original imshow)
    px.imshow(et_utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

def line(
    tensor, 
    renderer=None, 
    show=True,
    xticks=None,
    **kwargs
):
    xticks = xticks or range(tensor.shape[-1])
    xticks = [str(x) for x in xticks]
    if len(xticks) != len(set(xticks)):
        xticks = [f"{i}_{x}" for i, x in enumerate(xticks)]
    fig = px.line(y=et_utils.to_numpy(tensor), x=xticks, **kwargs)
    if show:
        fig.show(renderer)
    else:
        return fig


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = et_utils.to_numpy(x)
    y = et_utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def plot_topranks(batched_prompts, models=None, logits=None, labels=None, rank_cap=200):
    if logits is None:
        if models is None:
            raise ValueError("Pass either `models` or `logits`!")
        else:
            logits = [m(batched_prompts.clean_prompt, prepend_bos=True) for m in models]
    N_models = len(logits)
    if labels is None:
        labels = [f"Model {i}" for i in range(N_models)]
    series = []
    data = []
    N_prompts = 0
    for i in range(N_models):
        N_prompts = len(batched_prompts.correct_rank(logits[i]))
        correct_rank0_fraction = torch.sum((batched_prompts.correct_rank(logits[i])==0)/N_prompts)
        series.append([f"{labels[i]} ({correct_rank0_fraction.cpu():.0%} rank 0)"]*N_prompts)
        data.append(np.minimum(rank_cap, batched_prompts.correct_rank(logits[i]).cpu()))
    df = pd.DataFrame(dict(series=np.concatenate(series), 
                           data  =np.concatenate(data)))
    return px.histogram(df, x="data", color="series", barmode="overlay", nbins=rank_cap+1, labels={'data':f'Rank of correct answer (capped at {rank_cap})'}, title=f'Counts (out of {N_prompts})')

def plot_logitdiff(batched_prompts, models=None, logits=None, labels=None):
    if logits is None:
        if models is None:
            raise ValueError("Pass either `models` or `logits`!")
        else:
            logits = [m(batched_prompts.clean_prompt, prepend_bos=True) for m in models]
    N_models = len(logits)
    if labels is None:
        labels = [f"Model {i}" for i in range(N_models)]
    series = []
    data = []
    N_prompts = 0
    for i in range(N_models):
        N_prompts = len(batched_prompts.correct_rank(logits[i]))
        non_positive_fraction = torch.sum((batched_prompts.logit_diff(logits[i])<=0)/N_prompts)
        series.append([f"{labels[i]} ({non_positive_fraction.cpu():.0%} <= 0)"]*N_prompts)
        data.append(batched_prompts.logit_diff(logits[i]).cpu())
    df = pd.DataFrame(dict(series=np.concatenate(series), 
                           data  =np.concatenate(data)))
    return px.histogram(df, x="data", color="series", barmode="overlay", nbins=100, labels={'data':f'Logit diff'}, title=f'Counts (out of {N_prompts})')

def plot_residual_result(
    stream_result, 
    model, 
    labels=None, 
    title=None, 
    tokens_to=None, 
    batch_idx_labels=0, 
    show=True, 
    **kwargs
):
    if labels is None:
        labels = [f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(tokens_to[batch_idx_labels]))]
    fig = imshow(
        stream_result, 
        xticks=labels, 
        yticks=[*[f"before layer {n}" for n in range(model.cfg.n_layers)], f"after layer {model.cfg.n_layers-1}"], 
        title=title, 
        xlabel="position", 
        ylabel="layer", 
        show=False, 
        **kwargs
    )
    fig.update_xaxes(tickangle=90)
    fig.update_layout(
        font=dict(size=16),
        xaxis_tickfont=dict(family='monospace', size=17),
    )
    if show:
        fig.show()
    else:
        return fig

def plot_head_position_result(head_position_result, n_layers, pos_labels, title, show=True, **kwargs):
    n_heads = head_position_result.shape[0] // n_layers
    n_pos = head_position_result.shape[1]
    layerhead_labels = [f"{l}.{h}" for l in range(n_layers) for h in range(n_heads)]
    fig = imshow(head_position_result, xticks=pos_labels, yticks=layerhead_labels, xlabel="position", ylabel="layer.head",
             title=title, show=False, **kwargs)

    def add_line(y):
        fig.add_shape(type="line", x0=-0.5, x1=n_pos-0.5,
                    y0=y, y1=y, line=dict(color="black", width=1))

    for layer in range(n_layers + 1):
        add_line(layer * n_heads - 0.5)
    fig.update_xaxes(tickangle=90)
    fig.update_layout(
        font=dict(size=16),
        xaxis_tickfont=dict(family='monospace', size=17),
    )
    if show:
        fig.show()
    else:
        return fig


def plot_head_result(result, model, tokens_to=None, title=None, **kwargs):
    head_labels = [f"Head_{i}" for i in range(model.cfg.n_heads)]
    layer_labels = ["layer_"+str(n) for n in range(model.cfg.n_layers)]
    return imshow(result, xticks=head_labels, yticks=layer_labels, title=title, xlabel="Head", ylabel="Layer", **kwargs)


# Using Neel's original composition scores, we don't use them anymore
def plot_composition_scores_plt(model, mode="K"):
    comp = model.all_composition_scores(mode)
    labels = []
    for layer in range(model.cfg.n_layers):
        labels += [f"{layer}.{head}" for head in range(model.cfg.n_heads)]
    fig, ax = plt.subplots(figsize=(10,10))
    side = comp.shape[0] * comp.shape[1]
    comp = comp.reshape(side, side)
    im = ax.imshow(comp.cpu().detach())
    ax.set_xticks(range(side))
    ax.set_xticklabels(labels)
    ax.set_xlabel(f"{mode}-side to head")
    ax.set_ylabel(f"from head")
    ax.set_yticks(range(side))
    ax.set_yticklabels(labels)
    fig.colorbar(im)
    return fig, ax

# Using Neel's original composition scores, we don't use them anymore
def plot_composition_scores_px(model, mode="K"):
    comp = model.all_composition_scores(mode)
    labels = []
    for layer in range(model.cfg.n_layers):
        labels += [f"{layer}.{head}" for head in range(model.cfg.n_heads)]
    side = comp.shape[0] * comp.shape[1]
    comp = comp.reshape(side, side)
    imshow(comp.cpu().detach(),
       xticks=labels, yticks=labels, xlabel=f"{mode}-side to head",
       ylabel=f"from head")


@typechecked
def plot_any_composition_scores(scores: TT["n_heads", "layers_from", "n_heads"], title: str = None, zmin: float = None, zmax: float = None):
    n_heads, layer_to, _ = scores.shape
    xticks = [f"{layer_from}.{head_from}" for layer_from in range(layer_to) for head_from in range(n_heads)]
    yticks = [f"{layer_to}.{head_to}" for head_to in range(n_heads)]
    imshow(
        scores.view(n_heads, layer_to * n_heads),
        xticks=xticks,
        yticks=yticks,
        xlabel="From",
        ylabel="To",
        title=title,
        zmin=zmin,
        zmax=zmax
    )


@typechecked
def plot_patching_comp_scores(patching_comp_scores: TT["n_heads", "layers_from", "n_heads"], title=None, zmin=-100, zmax=100):
    plot_any_composition_scores(100 * patching_comp_scores, title=title or "Patching composition scores", zmin=zmin, zmax=zmax)


@typechecked
def plot_ov_comp_scores(ov_comp_scores: TT["n_heads", "layers_from", "n_heads"], title=None):
    ov_comp_scores_normalized = ov_comp_scores - ov_comp_scores.median()
    ov_comp_scores_normalized[ov_comp_scores_normalized < 0] = 0
    plot_any_composition_scores(ov_comp_scores_normalized, title=title or "OV composition scores")

@typechecked
def plot_patching_comp_scores_by_layer_to(patching_comp_scores_by_layer_to: Dict[int, torch.Tensor], zmin=-100, zmax=100):
    for layer_to, patching_comp_scores in patching_comp_scores_by_layer_to.items():
        plot_patching_comp_scores(patching_comp_scores, title=f"Patching composition scores for {layer_to=}", zmin=zmin, zmax=zmax)


@typechecked
def plot_ov_comp_scores_by_layer_to_by_mode(ov_comp_scores_by_layer_to_by_mode: Dict[str, Dict[int, torch.Tensor]]):
    for mode, ov_comp_scores_by_layer_to in ov_comp_scores_by_layer_to_by_mode.items():
        for layer_to, ov_comp_scores in ov_comp_scores_by_layer_to.items():
            plot_ov_comp_scores(ov_comp_scores, title=f"{mode}-composition scores for {layer_to=}")
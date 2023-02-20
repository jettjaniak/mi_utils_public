from .ext_imports import *

@typechecked
def logit_diff(logits: TT["batch", "pos", "d_vocab"],
        correct_tokens: TT["batch", "n_correct"],
        wrong_tokens: TT["batch", "n_wrong"],
        pos: int = -1) -> TT["batch"]:
    pos_logits = logits[:, pos, :]
    max_correct, _ = torch.gather(pos_logits, index=correct_tokens, dim=1).max(dim=1)
    max_wrong, _ = torch.gather(pos_logits, index=wrong_tokens, dim=1).max(dim=1)
    return max_correct - max_wrong


def from_pretrained(model_name: str, **kwargs) -> EasyTransformer:
    # This breaks: from_pretrained('gpt2', fold_value_biases=False)
    model_kwargs = {
        "center_unembed": True,
        "center_writing_weights": True,
        "fold_ln": True,
        "refactor_factored_attn_matrices": True, #What does this do?
    }
    for k, v in kwargs:
        model_kwargs[k] = v
    return EasyTransformer.from_pretrained(model_name, **model_kwargs)

def cond_no_space(model, v1, v2):
    # Returns true if good
    joint_str_tokens = model.to_str_tokens(f"{v1}{v2}", prepend_bos=False)
    v1_str_tokens = model.to_str_tokens(v1, prepend_bos=False)
    v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
    return joint_str_tokens == v1_str_tokens + v2_str_tokens

def cond_with_space(model, v1, v2):
    # Returns true if good
    joint_str_tokens = model.to_str_tokens(f" {v1}{v2}", prepend_bos=False)
    v1_str_tokens = model.to_str_tokens(" " + v1, prepend_bos=False)
    v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
    return joint_str_tokens == v1_str_tokens + v2_str_tokens

#def gpu_mem():
#    smi_stdout = subprocess.run("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits".split(" "), capture_output=True, text=True).stdout
#    my_pid = os.getpid()
#    mem_free = int(subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits".split(" "), capture_output=True, text=True).stdout.strip())
#    for proc_row in smi_stdout.split("\n"):
#        if proc_row:
#            pid, mem = proc_row.split(", ")
#            if int(pid) == my_pid:
#                print(f"You're using {int(mem) / 1024:.1f} GiB ({mem_free / 1024:.1f} GiB free)")
#                break

def gpu_mem():
    smi_stdout = subprocess.run("nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader,nounits".split(" "), capture_output=True, text=True).stdout
    my_pid = os.getpid()
    mem_free = subprocess.run("nvidia-smi --query-gpu=memory.free,gpu_bus_id --format=csv,noheader,nounits".split(" "), capture_output=True, text=True).stdout.strip()
    mem_free = [i for i in mem_free.split("\n")]
    owner = lambda pid: subprocess.run(f"ps -u -p {pid} | grep {pid} | cut -d ' ' -f 1", shell=True, capture_output=True, text=True).stdout.strip()
    for proc_row in smi_stdout.split("\n"):
        if proc_row:
            pid, mem, gpu_id = proc_row.split(", ")
            pid = int(pid)
            if pid == my_pid:
                print(f"You ({pid=}) are using {int(mem) / 1024:<4.1f} GiB @ GPU #{gpu_id[-4:]}")
            else:
                print(f"{owner(pid)} ({pid=}) is using {int(mem) / 1024:<4.1f} GiB @ GPU #{gpu_id[-4:]}")
    for i,m in enumerate(mem_free):
        mem,gpu_id = m.split(", ")
        mem = int(mem)
        print(f"Mem free: {mem/1024:<4.1f} GiB @ GPU #{gpu_id[-4:]}")

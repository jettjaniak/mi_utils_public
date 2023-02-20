from .ext_imports import *
from .prompts import *
from .misc import *
from . import variables


def generate_splittable_tokens(model, vocab_min=500, vocab_max=20000):
    all_split_tokens = []
    split_tokens_by_prefix = defaultdict(list)
    split_tokens_by_suffix = defaultdict(list)
    # In theory we could find all words by interating through the dict
    for i in range(vocab_min, vocab_max):
        token_str = model.tokenizer.decode(i)
        # Must be lowercase
        if token_str.lower() != token_str:
            continue
        # Must begin with a space
        if token_str[0] != " ":
            continue
        # Must be alphabetical (w/o space)
        if not token_str[1:].isalpha():
            continue
        # Must be an english word (w/o space)
        if token_str[1:] not in english_words:
            continue
        token_parts = model.to_str_tokens(token_str[1:], prepend_bos=False)
        # Double check it's a single token
        test_tokenization = model.to_str_tokens(token_str, prepend_bos=False)
        assert len(test_tokenization) == 1, (i, test_tokenization)
        # Must split into exactly two parts (w/o space)
        if len(token_parts) != 2:
            continue
        all_split_tokens.append(token_str)
        prefix, suffix = token_parts
        split_tokens_by_prefix[prefix].append(suffix)
        split_tokens_by_suffix[suffix].append(prefix)
    return all_split_tokens, split_tokens_by_prefix, split_tokens_by_suffix

def split_token_generator_normal(model, source_words, filler_words=variables.variable_names, extra_args=0):
    # Corrupt tokens: Just have a different random split-token at the end.
    def cond_no_space(v1, v2):
        joint_str_tokens = model.to_str_tokens(f"{v1}{v2}", prepend_bos=False)
        v1_str_tokens = model.to_str_tokens(v1, prepend_bos=False)
        v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
        return joint_str_tokens == v1_str_tokens + v2_str_tokens
    def cond_with_space(v1, v2):
        joint_str_tokens = model.to_str_tokens(f" {v1}{v2}", prepend_bos=False)
        v1_str_tokens = model.to_str_tokens(" " + v1, prepend_bos=False)
        v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
        return joint_str_tokens == v1_str_tokens + v2_str_tokens
    st = random.choice(source_words)[1:]
    st2 = random.choice(source_words)[1:]
    name = random.choice(filler_words)
    var = random.choice(filler_words)
    while not (cond_no_space(st, var) and cond_with_space(st, var) and cond_no_space(st2, var) and cond_with_space(st2, var)):
        var = random.choice(filler_words)
    extra_text = ", "+", ".join([random.choice(filler_words) for _ in range(extra_args)]) if extra_args>0 else ""
    clean_prompt = f"""    def {name}(self, {st}{var}{extra_text}):
        print({st}"""
    correct_answers = [f"{var}"]
    wrong_answers = [name, st, "_", ")", ","]
    corrupt_prompt = f"""    def {name}(self, {st}{var}{extra_text}):
        print({st2}"""
    return Prompt(clean_prompt, corrupt_prompt, correct_answers, wrong_answers)

def split_token_generator_random_prefix(model, source_words, filler_words=variables.variable_names):
    # Corrupt tokens: Just have a different random split-token at the end.
    def cond_no_space(v1, v2):
        joint_str_tokens = model.to_str_tokens(f"{v1}{v2}", prepend_bos=False)
        v1_str_tokens = model.to_str_tokens(v1, prepend_bos=False)
        v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
        return joint_str_tokens == v1_str_tokens + v2_str_tokens
    def cond_with_space(v1, v2):
        joint_str_tokens = model.to_str_tokens(f" {v1}{v2}", prepend_bos=False)
        v1_str_tokens = model.to_str_tokens(" " + v1, prepend_bos=False)
        v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
        return joint_str_tokens == v1_str_tokens + v2_str_tokens
    st = random.choice(source_words)[1:]
    st2 = random.choice(source_words)[1:]
    st_suffix = model.to_str_tokens(st, prepend_bos=False)[1]
    random_prefix = random.choice(filler_words)
    while not cond_no_space(random_prefix, st_suffix):
        random_prefix = random.choice(filler_words)
    name = random.choice(filler_words)
    var = random.choice(filler_words)
    while not (cond_no_space(st, var) and cond_with_space(st, var) and cond_no_space(st2, var) and cond_with_space(st2, var)):
        var = random.choice(filler_words)
    clean_prompt = f"""    def {name}(self, {st}{var}):
        print({random_prefix}{st_suffix}"""
    correct_answers = [f"{var}"]
    wrong_answers = [name, st, "_", ")", ","]
    corrupt_prompt = f"""    def {name}(self, {st}{var}):
        print({st2}"""
    return Prompt(clean_prompt, corrupt_prompt, correct_answers, wrong_answers)

def non_split_token_generator(model, source_words, filler_words=variables.variable_names, extra_args=0):
    # Just to check the induction
    def cond_no_space(v1, v2):
        joint_str_tokens = model.to_str_tokens(f"{v1}{v2}", prepend_bos=False)
        v1_str_tokens = model.to_str_tokens(v1, prepend_bos=False)
        v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
        return joint_str_tokens == v1_str_tokens + v2_str_tokens
    def cond_with_space(v1, v2):
        joint_str_tokens = model.to_str_tokens(f" {v1}{v2}", prepend_bos=False)
        v1_str_tokens = model.to_str_tokens(" " + v1, prepend_bos=False)
        v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
        return joint_str_tokens == v1_str_tokens + v2_str_tokens
    st = random.choice(source_words)[1:]
    st2 = random.choice(source_words)[1:]
    name = random.choice(filler_words)
    var = random.choice(filler_words)
    while not (cond_no_space(st, var) and cond_with_space(st, var) and cond_no_space(st2, var) and cond_with_space(st2, var)):
        var = random.choice(filler_words)
    extra_text = ", "+", ".join([random.choice(filler_words) for _ in range(extra_args)]) if extra_args>0 else ""
    clean_prompt = f"""    def {name}(self, {st}{var}{extra_text}):
        print( {st}"""
    correct_answers = [f"{var}"]
    wrong_answers = [name, st, "_", ")", ","]
    corrupt_prompt = f"""    def {name}(self, {st}{var}{extra_text}):
        print( {st2}"""
    return Prompt(clean_prompt, corrupt_prompt, correct_answers, wrong_answers)


def split_token_generator_multivar(model, sts, filler_words=variables.variable_names, n_args=3,
    random_print_prefix=False, random_print_suffix=False):
    # Corrupt tokens: Just have a different random split-token at the end.
    def cond_no_space(v1, v2):
        # Returns true if good
        joint_str_tokens = model.to_str_tokens(f"{v1}{v2}", prepend_bos=False)
        v1_str_tokens = model.to_str_tokens(v1, prepend_bos=False)
        v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
        #print("Checked (nospace)", joint_str_tokens, v1_str_tokens + v2_str_tokens)
        return joint_str_tokens == v1_str_tokens + v2_str_tokens
    def cond_with_space(v1, v2):
        joint_str_tokens = model.to_str_tokens(f" {v1}{v2}", prepend_bos=False)
        v1_str_tokens = model.to_str_tokens(" " + v1, prepend_bos=False)
        v2_str_tokens = model.to_str_tokens(v2, prepend_bos=False)
        #print("Checked (w/space)", joint_str_tokens, v1_str_tokens + v2_str_tokens)
        return joint_str_tokens == v1_str_tokens + v2_str_tokens
    sts_sample = random.sample(sts, n_args)
    args = random.sample(filler_words, n_args)
    #print("Sampled", sts, args)
    #[random.choice(filler_words) for _ in range(n_args)]
    def check_conds(sts, args):
        for i in range(n_args):
            for j in range(n_args):
                st = sts[i]
                var = args[j]
                if not (cond_no_space(st, var) and cond_with_space(st, var)):
                    return False
        return True
    name = random.choice(filler_words)
    while not check_conds(sts_sample, args):
        args = random.sample(filler_words, n_args)
    arglist = ", ".join([f"{st}{var}" for st, var in zip(sts_sample, args)])
    sts_index = random.randint(0, len(sts_sample)-1)
    st_print = sts_sample[sts_index]
    if random_print_prefix:
        st_print_prefix = model.to_str_tokens(random.choice(sts), prepend_bos=False)[0]
        st_print_suffix = model.to_str_tokens(st_print, prepend_bos=False)[1]
        while not cond_no_space(st_print_prefix, st_print_suffix):
            st_print_prefix = model.to_str_tokens(random.choice(sts), prepend_bos=False)[0]
            st_print_suffix = model.to_str_tokens(st_print, prepend_bos=False)[1]
        st_print = st_print_prefix+st_print_suffix
    if random_print_suffix:
        st_print_prefix = model.to_str_tokens(st_print, prepend_bos=False)[0]
        st_print_suffix = model.to_str_tokens(random.choice(sts), prepend_bos=False)[1]
        while not cond_no_space(st_print_prefix, st_print_suffix):
            st_print_prefix = model.to_str_tokens(st_print, prepend_bos=False)[0]
            st_print_suffix = model.to_str_tokens(random.choice(sts), prepend_bos=False)[1]
        st_print = st_print_prefix+st_print_suffix
    assert not (random_print_suffix and random_print_prefix)
    correct_answers = [args[sts_index]]
    clean_prompt = f"""    def {name}(self, {arglist}):
        print({st_print}"""
    st_print_corrupt = sts_sample[(sts_index+1)%len(sts_sample)]
    wrong_answers = copy(args)
    wrong_answers.remove(args[sts_index])
    corrupt_prompt = f"""    def {name}(self, {arglist}):
        print({st_print_corrupt}"""

    return Prompt(clean_prompt, corrupt_prompt, correct_answers, wrong_answers)



def split_token_generator_multivar_dict(model, sts, filler_words=variables.variable_names,
        n_induction_targets=3, random_print_prefix=False, random_print_suffix=False,
        return_target_index=False):
    # Select function name
    name = random.choice(filler_words)
    # Sample split tokens and following induction targets
    split_token_samples = random.sample(sts, n_induction_targets)
    induction_targets = random.sample(filler_words, n_induction_targets)
    def check_conds(split_tokens, follow_words):
        for i in range(n_induction_targets):
            for j in range(n_induction_targets):
                st = split_tokens[i]
                word = follow_words[j]
                if not (cond_no_space(model, st, word) and cond_with_space(model, st, word)):
                    return False
        return True
    while not check_conds(split_token_samples, induction_targets):
        induction_targets = random.sample(filler_words, n_induction_targets)
    # Select which one will be the answer
    ## Select one
    answer_index = random.randint(0, len(split_token_samples)-1)
    st_print = split_token_samples[answer_index]
    ## Random answer st (i.e. chance)
    if random_print_suffix and random_print_prefix:
        st_print = model.to_str_tokens(random.choice(sts), prepend_bos=False)
    ## Randomize prefix
    elif random_print_prefix:
        st_print_prefix = model.to_str_tokens(random.choice(sts), prepend_bos=False)[0]
        st_print_suffix = model.to_str_tokens(st_print, prepend_bos=False)[1]
        while not cond_no_space(model, st_print_prefix, st_print_suffix):
            st_print_prefix = model.to_str_tokens(random.choice(sts), prepend_bos=False)[0]
            st_print_suffix = model.to_str_tokens(st_print, prepend_bos=False)[1]
        st_print = st_print_prefix+st_print_suffix
    ## Randomize suffix
    elif random_print_suffix:
        st_print_prefix = model.to_str_tokens(st_print, prepend_bos=False)[0]
        st_print_suffix = model.to_str_tokens(random.choice(sts), prepend_bos=False)[1]
        while not cond_no_space(model, st_print_prefix, st_print_suffix):
            st_print_prefix = model.to_str_tokens(st_print, prepend_bos=False)[0]
            st_print_suffix = model.to_str_tokens(random.choice(sts), prepend_bos=False)[1]
        st_print = st_print_prefix+st_print_suffix
    # Make strings for result
    arglist = ", ".join([f"{st}{var}" for st, var in zip(split_token_samples, induction_targets)])
    # Make Prompt()
    correct_answers = [induction_targets[answer_index]]
    wrong_answers = copy(induction_targets)
    wrong_answers.remove(induction_targets[answer_index])
    clean_prompt = f"""    def {name}(self, {arglist}):
        print({st_print}"""
    st_print_corrupt = split_token_samples[(answer_index+1)%len(split_token_samples)]
    # Make corrupt prompts
    corrupt_prompt = {}
    ## Corrupt prompt with different word at end
    corrupt_prompt["different_end_word"] = f"""    def {name}(self, {arglist}):
        print({st_print_corrupt}"""
    ## Corrupt prompt with different key st
    extra_st = random.choice(sts)    
    extra_arg = correct_answers[0]
    while not (cond_no_space(model, extra_st, extra_arg) and cond_with_space(model, extra_st, extra_arg)):
        extra_st = random.choice(sts)
    split_token_samples_swap_relevant = copy(split_token_samples)
    split_token_samples_swap_relevant[answer_index] = extra_st
    arglist_swap_relevant_st = ", ".join([f"{st}{var}" for st, var in zip(split_token_samples_swap_relevant, induction_targets)])
    corrupt_prompt["change_relevant_st"] = f"""    def {name}(self, {arglist_swap_relevant_st}):
        print({st_print}"""
    ## Corrupt prompt with different irrelevant st
    sts_swap_index = random.randint(0, len(split_token_samples)-1)
    while sts_swap_index == answer_index:
        sts_swap_index = random.randint(0, len(split_token_samples)-1)
    extra_st = random.choice(sts)
    extra_arg = induction_targets[sts_swap_index]
    while not (cond_no_space(model, extra_st, extra_arg) and cond_with_space(model, extra_st, extra_arg)):
        sts_swap_index = random.randint(0, len(split_token_samples)-1)
        while sts_swap_index == answer_index:
            sts_swap_index = random.randint(0, len(split_token_samples)-1)
        extra_st = random.choice(sts)
        extra_arg = induction_targets[sts_swap_index]
    split_token_samples_swap_irrelevant = copy(split_token_samples)
    split_token_samples_swap_irrelevant[sts_swap_index] = extra_st
    arglist_swap_irrelevant_st = ", ".join([f"{st}{var}" for st, var in zip(split_token_samples_swap_irrelevant, induction_targets)])
    corrupt_prompt["change_irrelevant_st"] = f"""    def {name}(self, {arglist_swap_irrelevant_st}):
        print({st_print}"""
    # Corrupt prompt with different key arg
    P = Prompt(clean_prompt, corrupt_prompt, correct_answers, wrong_answers)
    if return_target_index:
        return P, sts_swap_index
    else:
        return P

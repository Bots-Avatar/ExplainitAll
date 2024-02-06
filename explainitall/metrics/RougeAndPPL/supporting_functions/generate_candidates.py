
'''
def generate_candidates(model, tokenizer, sentences, max_length=128):
    if tokenizer.pad_token is not None:
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        res = model.generate(**encoded_input, pad_token_id=0)
        candidates = tokenizer.batch_decode(res, skip_special_tokens=True)
        return candidates
    else:
        candidates = []
        for sentence in sentences:
            encoded_input = tokenizer(sentence, truncation=True, max_length=max_length, return_tensors='pt')
            res = model.generate(**encoded_input, pad_token_id=0)
            candidate = tokenizer.decode(res[0], skip_special_tokens=True)
            candidates.append(candidate)
        return candidates
'''

def generate_candidates(model, tokenizer, sentences, max_length=128, max_new_tokens=100):
    candidates = []
    for sentence in sentences:
        encoded_input = tokenizer(sentence, truncation=True, max_length=max_length, return_tensors='pt')
        encoded_input = encoded_input.to(model.device)
        res = model.generate(**encoded_input, max_new_tokens=max_new_tokens)
        candidate = tokenizer.decode(res[0], skip_special_tokens=True)
        candidates.append(candidate)
    return candidates

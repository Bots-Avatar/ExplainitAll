import numpy as np


class GPT2HMMDataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_data_1(self, texts):
        list_y = []
        for text in texts:
            list_tok = [1, 1] + self.tokenizer(text)['input_ids'] + [2]
            list_y.append(np.array(list_tok))
        return np.concatenate(list_y)

    @staticmethod
    def token_pair_2_str(tokens):
        list_x = list(tokens)
        return str(list_x)

    def createXY(self, tokens_gpt):
        x = []
        y = []

        for i in range(len(tokens_gpt) - 2):
            str_x = self.token_pair_2_str(tokens_gpt[i:i + 2])
            x.append(str_x)
            y.append(tokens_gpt[i + 2])

        return x, y

    @staticmethod
    def encode_samples_x(x, y_encode):
        return [y_encode[i] for i in x]

    @staticmethod
    def encode_samples_y(y, y_encode):
        return [y_encode[i] for i in y]

    @staticmethod
    def encode_end(tokens_gpt, x_encode):
        str_x = GPT2HMMDataProcessor.token_pair_2_str(tokens_gpt[-2:])
        return x_encode[str_x] if str_x in x_encode else -1

    def create_data(self, tokens_gpt):
        x, y = self.createXY(tokens_gpt)
        x_set = list(set(x))
        mask_y = list(set(y))

        x_encode = {}
        x_decode = []

        y_encode = {}
        y_decode = []

        for i, xi in enumerate(x_set):
            x_encode.update({xi: i})
            x_decode.append(xi)

        for i, yi in enumerate(mask_y):
            y_encode.update({yi: i})
            y_decode.append(yi)

        x_enc = self.encode_samples_x(x, x_encode)
        y_enc = self.encode_samples_y(y, y_encode)

        return {'x': x_enc, 'y': y_enc, 'x_encoder': x_encode, 'y_encoder': y_encode, 'x_decoder': x_decode,
                'y_decoder': y_decode}

    @staticmethod
    def train(data):
        states = {}
        n = len(data['y'])
        x = data['x']
        y = data['y']

        for i in range(n):
            if x[i] in states:
                if y[i] in states[x[i]]['tokens']:
                    states[x[i]]['probs'][y[i]] += 1
                else:
                    states[x[i]]['probs'].update({y[i]: 1})
                    states[x[i]]['tokens'].update({y[i]: 0})
            else:
                states.update({x[i]: {'tokens': {}, 'probs': {}}})
                states[x[i]]['probs'].update({y[i]: 1})
                states[x[i]]['tokens'].update({y[i]: 0})

        n = max(states.keys()) + 1
        n_states = []

        for i in range(n):
            tokens = list(states[i]['tokens'].keys())
            total_count = 0
            probs = []

            for t in tokens:
                count = states[i]['probs'][t]
                total_count += count
                probs.append(count)

            for j, p in enumerate(probs):
                probs[j] = p / total_count

            n_states.append({'tokens': tokens, 'probs': probs})

        return n_states

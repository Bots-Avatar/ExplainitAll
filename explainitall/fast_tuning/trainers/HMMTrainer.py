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
        X = []
        Y = []

        for i in range(len(tokens_gpt) - 2):
            str_x = self.token_pair_2_str(tokens_gpt[i:i + 2])
            X.append(str_x)
            Y.append(tokens_gpt[i + 2])

        return X, Y

    @staticmethod
    def encode_samples_x(X, x_encode):
        encoded = []
        for x in X:
            encoded.append(x_encode[x])
        return encoded

    @staticmethod
    def encode_samples_y(Y, y_encode):
        encoded = []
        for y in Y:
            encoded.append(y_encode[y])
        return encoded

    @staticmethod
    def encode_end(tokens_gpt, x_encode):
        str_x = GPT2HMMDataProcessor.token_pair_2_str(tokens_gpt[-2:])
        return x_encode[str_x] if str_x in x_encode else -1

    def create_data(self, tokens_gpt):
        X, Y = self.createXY(tokens_gpt)
        X_set = list(set(X))
        mask_y = list(set(Y))

        x_encode = {}
        x_decode = []

        y_encode = {}
        y_decode = []

        for i, x in enumerate(X_set):
            x_encode.update({x: i})
            x_decode.append(x)

        for i, y in enumerate(mask_y):
            y_encode.update({y: i})
            y_decode.append(y)

        x_enc = self.encode_samples_x(X, x_encode)
        y_enc = self.encode_samples_y(Y, y_encode)

        return {'x': x_enc, 'y': y_enc, 'x_encoder': x_encode, 'y_encoder': y_encode, 'x_decoder': x_decode,
                'y_decoder': y_decode}

    @staticmethod
    def train(data):
        states = {}
        N = len(data['y'])
        x = data['x']
        y = data['y']

        for i in range(N):
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

        N = max(states.keys()) + 1
        n_states = []

        for i in range(N):
            tokens = list(states[i]['tokens'].keys())
            n = 0
            probs = []

            for t in tokens:
                p = states[i]['probs'][t]
                n += p
                probs.append(p)

            for i, p in enumerate(probs):
                probs[i] = p / n

            n_states.append({'tokens': tokens, 'probs': probs})

        return n_states

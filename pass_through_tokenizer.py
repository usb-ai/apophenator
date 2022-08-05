class PassThroughTokenizer:

    def __init__(self):
        pass

    def __call__(self, sentence, *args, **kwargs):
        return sentence  # list of tokens
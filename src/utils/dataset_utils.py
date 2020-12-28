class Token:
    def __init__(self, start, text, tag, pos_tag=None, dep_tag=None, guidance_tag=None):
        self.start = start
        self.text = text
        self.tag = tag
        self.pos_tag = pos_tag
        self.dep_tag = dep_tag
        self.guidance_tag = guidance_tag

    def __str__(self):
        return "({0}, {1}, {2}, {3}, {4}, {5})".format(self.start, self.text, self.tag, self.pos_tag, self.dep_tag,
                                                       self.guidance_tag)

    def __repr__(self):
        return "({0}, {1}, {2}, {3}, {4}, {5})".format(self.start, self.text, self.tag, self.pos_tag, self.dep_tag,
                                                       self.guidance_tag)


class BertToken(Token):
    def __init__(self, start, text, bert_id, tag, pos_tag=None, dep_tag=None, guidance_tag=None):
        super(BertToken, self).__init__(start, text, tag, pos_tag, dep_tag, guidance_tag)
        self.bert_id = bert_id

    def __str__(self):
        return "({0}, {1}, {2}, {3}, {4}, {5}, {6})".format(self.start, self.text, self.bert_id, self.tag, self.pos_tag,
                                                            self.dep_tag, self.guidance_tag)

    def __repr__(self):
        return "({0}, {1}, {2}, {3}, {4}, {5}, {6})".format(self.start, self.text, self.bert_id, self.tag, self.pos_tag,
                                                            self.dep_tag, self.guidance_tag)


class Entity:
    def __init__(self, start_token, end_token, text, tag):
        self.start_token = start_token
        self.end_token = end_token
        self.text = text
        self.tag = tag

    def __str__(self):
        return "({0}, {1}, {2}, {3})".format(self.start_token, self.end_token, self.text, self.tag)

    def __repr__(self):
        return "({0}, {1}, {2}, {3})".format(self.start_token, self.end_token, self.text, self.tag)


class Sentence:
    def __init__(self, tokens, entities):
        self.tokens = tokens
        self.entities = entities

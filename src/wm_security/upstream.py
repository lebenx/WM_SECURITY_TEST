"""Bind our logged client to WorldEvolver's original text-model API."""
from . import native  # Registers the pinned upstream source directory.
from world_model.episodic_semantic import WMEpisodicSemantic


class LoggedLLM:
    def __init__(self, client, prefix):
        self.client = client
        self.prefix = prefix
        self.tag = 'call'
        self.max_tokens = 1536
        self.stop = None
        self.counter = 0
        self.context_length = 32768

    def generate_raw(self, system, user):
        self.counter += 1
        result = self.client.chat([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ], max_tokens=self.max_tokens, tag=f'{self.prefix}/{self.tag}/{self.counter}')
        return True, result

    generate = generate_raw

    def num_tokens_from_messages(self, messages):
        tokenizer = getattr(self.client, 'tokenizer', None)
        if tokenizer is not None:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
            return len(tokenizer.encode(rendered, add_special_tokens=False))
        # Conservative bound for API-only clients; no history is truncated in
        # the small decision probes. Model token usage remains in call logs.
        return sum(len(m['content'].encode('utf-8')) + 8 for m in messages)


def text_world_model(client, prefix):
    llm = LoggedLLM(client, prefix)
    return WMEpisodicSemantic(llm=llm, env_name='scienceworld', top_k=5,
                             batch_k=1, token_budget=2048, sf_confidence_pct=0,
                             wm_max_tokens=1536)

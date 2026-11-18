from functools import lru_cache

import ollama


@lru_cache(maxsize=1)
def get_ollama_client() -> ollama.Client:
    return ollama.Client(trust_env=False)

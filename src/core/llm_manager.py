from abc import ABC, abstractmethod
from typing import List, Optional
import tiktoken

class LLMManager(ABC):
    def __init__(self, model_name: str = "gpt-3.5-turbo", context_limit: int = 3000):
        self.model_name = model_name
        self.context_limit = context_limit
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        pass

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def truncate_context(self, context_list: List[str], buffer: int = 500) -> str:
        """Truncate context strings to fit within the token limit."""
        target_limit = self.context_limit - buffer
        truncated_context = []
        current_tokens = 0
        
        for ctx in context_list:
            ctx_tokens = self.count_tokens(ctx)
            if current_tokens + ctx_tokens <= target_limit:
                truncated_context.append(ctx)
                current_tokens += ctx_tokens
            else:
                remaining_tokens = target_limit - current_tokens
                if remaining_tokens > 10:
                    # Partial truncation of the last useful chunk
                    tokens = self.tokenizer.encode(ctx)
                    truncated_context.append(self.tokenizer.decode(tokens[:remaining_tokens]))
                break
                
        return "\n\n".join(truncated_context)

class MockLLMManager(LLMManager):
    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        print(f"Generating with MockLLM... Prompt length: {len(prompt)}")
        return "This is a simulated response based on the provided context."

# Concrete Ollama or OpenAI managers can be added here

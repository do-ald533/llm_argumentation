"""LLM client wrapper using LangChain."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import openai
from typing import Optional
import time
from src.config import Config
from src.logging_config import get_logger


class LLMClient:
    """Unified LLM client supporting OpenAI and DeepSeek."""
    
    def __init__(self, config: Config):
        """Initialize LLM client.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.model_name = config.openai_model
        self.logger = get_logger("llm_client")
        
        # Use native OpenAI for more control
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.parser = StrOutputParser()
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: Optional[float] = None
    ) -> str:
        """Generate completion from prompt.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (defaults to config)
            
        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.config.temperature
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    timeout=self.config.timeout_seconds
                )
                return response.choices[0].message.content.strip()
            
            except openai.RateLimitError as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning("rate_limit_hit", wait_time=wait_time, attempt=attempt)
                    time.sleep(wait_time)
                else:
                    raise
            
            except openai.APIError as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning("api_error_retry", error=str(e), wait_time=wait_time, attempt=attempt)
                    time.sleep(wait_time)
                else:
                    raise
            
            except Exception as e:
                self.logger.error("unexpected_llm_error", error=str(e), exc_info=True)
                raise
        
        raise RuntimeError("Failed to generate completion after all retries")
    
    def generate_with_thinking(self, prompt: str, **kwargs) -> str:
        """Generate completion and extract content after </think> tag.
        
        Args:
            prompt: User prompt
            **kwargs: Additional arguments for generate()
            
        Returns:
            Generated text after </think> tag
        """
        output = self.generate(prompt, **kwargs)
        return self._extract_after_think(output)
    
    @staticmethod
    def _extract_after_think(output: str) -> str:
        """Extract content after </think> tag."""
        if "</think>" in output:
            return output.split("</think>", 1)[-1].strip()
        return output

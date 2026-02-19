"""LLM client wrapper using OpenAI."""
import openai
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
import time
from src.config import Config
from src.logging_config import get_logger

T = TypeVar('T', bound=BaseModel)


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
        
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate completion from prompt.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
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
    
    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> T:
        """Generate structured output using OpenAI's structured output feature.
        
        Args:
            prompt: User prompt
            response_model: Pydantic model class for structured output
            system_message: Optional system message
            temperature: Temperature for generation (defaults to config)
            
        Returns:
            Parsed Pydantic model instance
        """
        temp = temperature if temperature is not None else self.config.temperature
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.config.max_retries):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_model,
                    temperature=temp,
                    timeout=self.config.timeout_seconds
                )
                return completion.choices[0].message.parsed
            
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
                self.logger.error("unexpected_llm_error_structured", error=str(e), exc_info=True)
                raise
        
        raise RuntimeError("Failed to generate structured completion after all retries")

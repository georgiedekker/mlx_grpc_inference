"""
MLX Inference Module

This module provides a high-level interface for text generation using MLX models.
It supports both simple text generation and chat-based interactions with
configurable parameters for temperature, top_p, and repetition penalty.
"""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
from typing import List, Dict, Optional, Union, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class MLXInferenceError(Exception):
    """Custom exception for MLX inference errors."""
    pass


class MLXInference:
    """
    A class for managing MLX model inference.
    
    This class handles model loading and provides methods for text generation
    with support for chat templates and various sampling parameters.
    
    Attributes:
        model_name (str): The name/path of the MLX model to load
        model: The loaded MLX model
        tokenizer: The tokenizer associated with the model
    """
    
    def __init__(self, model_name: str = "mlx-community/Qwen3-1.7B-8bit"):
        """
        Initialize the MLX inference engine.
        
        Args:
            model_name (str): The name or path of the MLX model to load.
                            Defaults to "mlx-community/Qwen3-1.7B-8bit".
                            
        Raises:
            MLXInferenceError: If the model fails to load.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        try:
            logger.info(f"Loading model: {model_name}")
            self.model, self.tokenizer = load(model_name)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load model '{model_name}': {str(e)}"
            logger.error(error_msg)
            raise MLXInferenceError(error_msg) from e
    
    def _validate_generation_params(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float
    ) -> None:
        """
        Validate generation parameters.
        
        Args:
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty factor
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
        
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {top_p}")
        
        if repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")
    
    def _create_sampling_config(
        self,
        temperature: float,
        top_p: float,
        repetition_penalty: float
    ) -> tuple[Callable, list[Callable]]:
        """
        Create sampler and logits processors for generation.
        
        Args:
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty factor
            
        Returns:
            Tuple of (sampler, logits_processors)
        """
        sampler = make_sampler(temp=temperature, top_p=top_p)
        
        logits_processors = []
        if repetition_penalty > 1.0:
            logits_processors.append(make_repetition_penalty(repetition_penalty))
        
        return sampler, logits_processors
    
    def _generate(
        self,
        formatted_prompt: Union[str, List[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        verbose: bool
    ) -> Tuple[str, int]:
        """
        Internal method to generate text from a formatted prompt.
        
        This method handles the actual generation logic that's common to both
        generate_response and chat methods.
        
        Args:
            formatted_prompt: The pre-formatted prompt (string or token list)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty factor
            verbose: Whether to print generation info
            
        Returns:
            Tuple of (generated text, token count)
            
        Raises:
            MLXInferenceError: If generation fails
        """
        # Check if prompt is empty - handle both string and list cases
        if isinstance(formatted_prompt, str) and not formatted_prompt.strip():
            raise ValueError("Formatted prompt cannot be empty")
        elif isinstance(formatted_prompt, list) and not formatted_prompt:
            raise ValueError("Formatted prompt cannot be empty")
        
        try:
            self._validate_generation_params(max_tokens, temperature, top_p, repetition_penalty)
            
            sampler, logits_processors = self._create_sampling_config(
                temperature, top_p, repetition_penalty
            )
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors if logits_processors else None,
                verbose=verbose
            )
            
            # Count tokens in the response
            # The tokenizer encode method returns token IDs
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            token_count = len(response_tokens)
            
            return response, token_count
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            raise MLXInferenceError(error_msg) from e
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        verbose: bool = False,
        return_token_count: bool = False
    ) -> Union[str, Tuple[str, int]]:
        """
        Generate a text response from a single prompt.
        
        This method is suitable for simple text generation tasks where you have
        a single prompt string and want to generate a continuation.
        
        Args:
            prompt (str): The input prompt text
            max_tokens (int): Maximum number of tokens to generate (default: 512)
            temperature (float): Controls randomness in generation. Higher values
                               make output more random (default: 0.7, range: 0.0-2.0)
            top_p (float): Nucleus sampling parameter. Only tokens with cumulative
                          probability up to top_p are considered (default: 0.9, range: 0.0-1.0)
            repetition_penalty (float): Penalty for repeating tokens. Higher values
                                      reduce repetition (default: 1.1, min: 1.0)
            verbose (bool): If True, print generation statistics (default: False)
            return_token_count (bool): If True, return a tuple of (response, token_count) (default: False)
            
        Returns:
            Union[str, Tuple[str, int]]: The generated text response, or tuple of (response, token_count)
            
        Raises:
            ValueError: If prompt is empty or parameters are invalid
            MLXInferenceError: If generation fails
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Format prompt with chat template if available
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        response, token_count = self._generate(
            formatted_prompt, max_tokens, temperature, top_p, repetition_penalty, verbose
        )
        
        if return_token_count:
            return response, token_count
        return response
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        verbose: bool = False,
        return_token_count: bool = False
    ) -> Union[str, Tuple[str, int]]:
        """
        Generate a response in a chat conversation context.
        
        This method is designed for chat-based interactions where you have a
        conversation history represented as a list of messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries, each with
                                           'role' and 'content' keys. Role can be
                                           'user', 'assistant', or 'system'.
            max_tokens (int): Maximum number of tokens to generate (default: 512)
            temperature (float): Controls randomness in generation (default: 0.7, range: 0.0-2.0)
            top_p (float): Nucleus sampling parameter (default: 0.9, range: 0.0-1.0)
            repetition_penalty (float): Penalty for repeating tokens (default: 1.1, min: 1.0)
            verbose (bool): If True, print generation statistics (default: False)
            return_token_count (bool): If True, return a tuple of (response, token_count) (default: False)
            
        Returns:
            Union[str, Tuple[str, int]]: The generated response, or tuple of (response, token_count)
            
        Raises:
            ValueError: If messages is empty, invalid format, or parameters are invalid
            MLXInferenceError: If generation fails
            
        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi! How can I help you?"},
            ...     {"role": "user", "content": "What's the weather like?"}
            ... ]
            >>> response = model.chat(messages)
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Validate message format
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message at index {i} must be a dictionary")
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message at index {i} must have 'role' and 'content' keys")
            if msg["role"] not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role '{msg['role']}' at index {i}")
            if not msg["content"].strip():
                raise ValueError(f"Message content at index {i} cannot be empty")
        
        # Format messages using chat template if available
        if self.tokenizer.chat_template is not None:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            # Fallback formatting for models without chat templates
            formatted_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            formatted_prompt += "\nassistant: "  # Add prompt for assistant response
        
        response, token_count = self._generate(
            formatted_prompt, max_tokens, temperature, top_p, repetition_penalty, verbose
        )
        
        if return_token_count:
            return response, token_count
        return response
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is successfully loaded."""
        return self.model is not None and self.tokenizer is not None
    
    def __repr__(self) -> str:
        """String representation of the MLXInference instance."""
        status = "loaded" if self.is_loaded else "not loaded"
        return f"MLXInference(model_name='{self.model_name}', status='{status}')"
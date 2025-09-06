import logging
import torch
from transformers.pipelines import pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    BartForConditionalGeneration,
    BartTokenizer
)
from typing import Optional, Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML models for summarization and question answering"""

    def __init__(self):
        self.summarization_models = {
            'bart-large-cnn': 'facebook/bart-large-cnn',
            'distilbart-cnn': 'sshleifer/distilbart-cnn-12-6',
            'pegasus-xsum': 'google/pegasus-xsum',
            't5-small': 't5-small'
        }

        self.qa_models = {
            'roberta-squad2': 'deepset/roberta-base-squad2',
            'distilbert-squad': 'distilbert-base-uncased-distilled-squad',
            'bert-squad2': 'deepset/bert-base-cased-squad2'
        }

        self._summarizer = None
        self._qa_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @st.cache_resource
    def load_summarization_model(_self, model_name: str = 'bart-large-cnn') -> pipeline:
        """
        Load and cache summarization model

        Args:
            model_name: Name of the model to load

        Returns:
            Summarization pipeline
        """
        try:
            model_path = _self.summarization_models.get(model_name, _self.summarization_models['bart-large-cnn'])
            logger.info(f"Loading summarization model: {model_path}")

            # Load model with optimizations
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )

            summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            logger.info(f"Successfully loaded summarization model: {model_path}")
            return summarizer

        except Exception as e:
            logger.error(f"Error loading summarization model {model_name}: {e}")
            # Fallback to smaller model
            try:
                logger.info("Attempting to load fallback model: distilbart-cnn")
                return pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as fallback_error:
                logger.error(f"Fallback model failed: {fallback_error}")
                raise Exception(f"Failed to load any summarization model: {str(e)}")

    @st.cache_resource
    def load_qa_model(_self, model_name: str = 'roberta-squad2') -> pipeline:
        """
        Load and cache question answering model

        Args:
            model_name: Name of the model to load

        Returns:
            Q&A pipeline
        """
        try:
            model_path = _self.qa_models.get(model_name, _self.qa_models['roberta-squad2'])
            logger.info(f"Loading Q&A model: {model_path}")

            qa_pipeline = pipeline(
                "question-answering",
                model=model_path,
                tokenizer=model_path,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )

            logger.info(f"Successfully loaded Q&A model: {model_path}")
            return qa_pipeline

        except Exception as e:
            logger.error(f"Error loading Q&A model {model_name}: {e}")
            # Fallback to distilbert
            try:
                logger.info("Attempting to load fallback Q&A model: distilbert-squad")
                return pipeline(
                    "question-answering",
                    model="distilbert-base-uncased-distilled-squad",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as fallback_error:
                logger.error(f"Fallback Q&A model failed: {fallback_error}")
                raise Exception(f"Failed to load any Q&A model: {str(e)}")

    def summarize_text(self, text: str, summarizer: pipeline,
                      max_length: int = 150, min_length: int = 30,
                      do_sample: bool = False) -> Dict[str, Any]:
        """
        Generate summary using the loaded model

        Args:
            text: Text to summarize
            summarizer: Summarization pipeline
            max_length: Maximum summary length
            min_length: Minimum summary length
            do_sample: Whether to use sampling

        Returns:
            Dictionary with summary and metadata
        """
        try:
            if not text or len(text.split()) < 20:
                return {
                    'summary': "Text is too short to summarize effectively.",
                    'success': False,
                    'error': "Insufficient text length"
                }

            # Ensure text length is appropriate for model
            max_input_length = 1024
            words = text.split()

            if len(words) > max_input_length:
                text = ' '.join(words[:max_input_length])
                logger.warning(f"Text truncated to {max_input_length} words")

            # Generate summary
            result = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                clean_up_tokenization_spaces=True,
                truncation=True
            )

            summary = result[0]['summary_text']

            return {
                'summary': summary,
                'success': True,
                'original_length': len(words),
                'summary_length': len(summary.split()),
                'compression_ratio': len(summary.split()) / len(words)
            }

        except Exception as e:
            logger.error(f"Error in text summarization: {e}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'success': False,
                'error': str(e)
            }

    def answer_question(self, question: str, context: str,
                       qa_pipeline: pipeline,
                       confidence_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Answer question using the loaded Q&A model

        Args:
            question: Question to answer
            context: Context text
            qa_pipeline: Q&A pipeline
            confidence_threshold: Minimum confidence score

        Returns:
            Dictionary with answer and metadata
        """
        try:
            if not question or not context:
                return {
                    'answer': "Please provide both a question and context.",
                    'success': False,
                    'error': "Missing question or context"
                }

            # Limit context length for model
            max_context_length = 2000
            if len(context) > max_context_length:
                # Try to keep relevant context around potential answer location
                context = context[:max_context_length]
                logger.warning(f"Context truncated to {max_context_length} characters")

            # Get answer from model
            result = qa_pipeline(question=question, context=context)

            if isinstance(result, list):
                result = result[0]  # Take the best answer

            confidence = result.get('score', 0)
            answer = result.get('answer', '')
            start_pos = result.get('start', 0)
            end_pos = result.get('end', 0)

            # Check confidence threshold
            if confidence < confidence_threshold:
                return {
                    'answer': "I'm not confident enough to answer this question based on the provided document. Please try rephrasing your question or check if the information is available in the document.",
                    'success': False,
                    'confidence': confidence,
                    'low_confidence': True
                }

            return {
                'answer': answer,
                'success': True,
                'confidence': confidence,
                'start_position': start_pos,
                'end_position': end_pos,
                'context_length': len(context)
            }

        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            return {
                'answer': f"Error processing question: {str(e)}",
                'success': False,
                'error': str(e)
            }

    def get_model_info(self, model_type: str) -> Dict[str, str]:
        """
        Get information about available models

        Args:
            model_type: 'summarization' or 'qa'

        Returns:
            Dictionary of model information
        """
        if model_type == 'summarization':
            return {
                'bart-large-cnn': 'BART Large CNN - High quality, slower',
                'distilbart-cnn': 'DistilBART CNN - Fast, good quality',
                'pegasus-xsum': 'Pegasus XSum - Good for news/articles',
                't5-small': 'T5 Small - Very fast, basic quality'
            }
        elif model_type == 'qa':
            return {
                'roberta-squad2': 'RoBERTa Squad2 - High accuracy',
                'distilbert-squad': 'DistilBERT Squad - Fast, good accuracy',
                'bert-squad2': 'BERT Squad2 - Good accuracy, slower'
            }
        else:
            return {}

    def clear_cache(self):
        """Clear model cache to free memory"""
        try:
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            self._summarizer = None
            self._qa_pipeline = None

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the device being used"""
        device_info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            device_info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(0),
                'cuda_memory_allocated': torch.cuda.memory_allocated(0),
                'cuda_memory_cached': torch.cuda.memory_reserved(0)
            })

        return device_info

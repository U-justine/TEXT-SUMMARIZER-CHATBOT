import re
import logging
from typing import List, Optional
import PyPDF2
from io import BytesIO

logger = logging.getLogger(__name__)

class TextProcessor:
    """Utility class for text processing operations"""

    def __init__(self):
        self.max_chunk_size = 1024
        self.min_summary_length = 30
        self.max_summary_length = 150

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from uploaded PDF file

        Args:
            pdf_file: Uploaded PDF file object

        Returns:
            str: Extracted and cleaned text
        """
        try:
            # Reset file pointer to beginning
            pdf_file.seek(0)

            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            # Clean the extracted text
            cleaned_text = self.clean_text(text)
            logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")

            return cleaned_text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text

        Args:
            text: Raw text to clean

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace and newlines
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()"-]', '', text)

        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)

        # Remove multiple consecutive punctuation marks
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def chunk_text(self, text: str, max_length: Optional[int] = None) -> List[str]:
        """
        Split text into chunks for processing

        Args:
            text: Text to chunk
            max_length: Maximum length per chunk (defaults to self.max_chunk_size)

        Returns:
            List[str]: List of text chunks
        """
        if max_length is None:
            max_length = self.max_chunk_size

        if not text:
            return []

        # Split by sentences first to maintain context
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence) + 1  # +1 for space

            # If adding this sentence would exceed max_length
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    def validate_text_for_processing(self, text: str) -> tuple[bool, str]:
        """
        Validate if text is suitable for processing

        Args:
            text: Text to validate

        Returns:
            tuple: (is_valid, message)
        """
        if not text or not text.strip():
            return False, "No text provided"

        word_count = len(text.split())

        if word_count < 20:
            return False, "Text is too short (minimum 20 words required)"

        if word_count > 50000:
            return False, "Text is too long (maximum 50,000 words allowed)"

        # Check for meaningful content (not just numbers or special characters)
        meaningful_chars = re.sub(r'[^\w\s]', '', text)
        if len(meaningful_chars) < len(text) * 0.5:
            return False, "Text contains too many special characters"

        return True, "Text is valid for processing"

    def get_text_statistics(self, text: str) -> dict:
        """
        Get statistics about the text

        Args:
            text: Text to analyze

        Returns:
            dict: Text statistics
        """
        if not text:
            return {
                'word_count': 0,
                'character_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'avg_words_per_sentence': 0
            }

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0

        return {
            'word_count': len(words),
            'character_count': len(text),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_words_per_sentence': round(avg_words_per_sentence, 2)
        }

    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract key phrases from text using simple frequency analysis

        Args:
            text: Text to analyze
            top_n: Number of top phrases to return

        Returns:
            List[str]: List of key phrases
        """
        if not text:
            return []

        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())

        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]

        # Count word frequencies
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top_n
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]

    def prepare_text_for_model(self, text: str, model_max_length: int = 1024) -> str:
        """
        Prepare text for model processing by truncating if necessary

        Args:
            text: Text to prepare
            model_max_length: Maximum length accepted by the model

        Returns:
            str: Prepared text
        """
        if not text:
            return ""

        # If text is within limits, return as is
        if len(text.split()) <= model_max_length:
            return text

        # Truncate to model max length, trying to end at sentence boundary
        words = text.split()
        truncated = ' '.join(words[:model_max_length])

        # Try to end at the last complete sentence
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )

        if last_sentence_end > len(truncated) * 0.8:  # If we can keep 80% of text
            truncated = truncated[:last_sentence_end + 1]

        return truncated

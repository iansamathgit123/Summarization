# src/model.py
# Main file containing the model class and demonstration function
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import yaml
import os
import logging

class DocumentationSummarizer:
    def __init__(self, config_path='../config/config.yaml'):
        """
        Initialize the Documentation Summarization Model
        
        Args:
            config_path (str): Path to configuration file
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path))
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
        
        # Initialize model and tokenizer
        self.model_name = self.config['model']['name']
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
        
        # Configure model parameters
        self.max_input_length = self.config['model']['max_input_length']
        self.max_output_length = self.config['model']['max_output_length']
    
    def generate_summary(self, text, **kwargs):
        """
        Generate abstractive summary for given text with flexible length parameters
        
        Args:
            text (str): Input documentation text
            **kwargs: Flexible keyword arguments for length control
        
        Returns:
            str: Generated summary
        """
        try:
            # Use configuration defaults with flexible overrides
            min_length = kwargs.get('min_length', 30)
            max_length = kwargs.get('max_length', self.max_output_length)
            
            # Preprocess input
            input_text = f"summarize: {text}"
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=self.max_input_length, 
                truncation=True
            )
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs,
                min_length=min_length,
                max_length=max_length,
                num_return_sequences=1,
                temperature=self.config['inference']['temperature'],
                top_k=self.config['inference']['top_k'],
                top_p=self.config['inference']['top_p']
            )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary due to an error."
    
    def fine_tune(self, training_data):
        """
        Fine-tune the model on specific documentation data
        
        Args:
            training_data (list): List of training examples
        
        Returns:
            dict: Fine-tuning metrics
        """
        try:
            # Prepare training inputs
            inputs = self.tokenizer(
                [f"summarize: {item['text']}" for item in training_data],
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            targets = self.tokenizer(
                [item['summary'] for item in training_data],
                max_length=self.max_output_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Placeholder for actual fine-tuning logic
            self.logger.info("Fine-tuning process initiated")
            
            return {
                "status": "Fine-tuning simulation",
                "total_samples": len(training_data)
            }
        
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            return {"status": "Error", "message": str(e)}

def main():
    """
    Demonstration of model usage
    """
    # Initialize the summarizer
    summarizer = DocumentationSummarizer()
    
    # Example documentation text
    sample_text = """
    Python is a high-level, interpreted programming language known for its 
    readability and versatility. It supports multiple programming paradigms, 
    including procedural, object-oriented, and functional programming. 
    Python is widely used in web development, data science, artificial intelligence, 
    and scientific computing.
    """
    
    # Generate summary with default length
    default_summary = summarizer.generate_summary(sample_text)
    print("Default Summary:")
    print(default_summary)
    
    # Generate summary with custom length
    custom_summary = summarizer.generate_summary(
        sample_text, 
        min_length=20, 
        max_length=50
    )
    print("\nCustom Length Summary:")
    print(custom_summary)

if __name__ == "__main__":
    main()
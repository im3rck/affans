"""
Fine-tuning Module for Customer Support Model
Supports both OpenAI fine-tuning and Hugging Face LoRA fine-tuning
"""

import os
import json
from typing import List, Dict, Optional
import openai
from openai import OpenAI
import time
from pathlib import Path


class OpenAIFineTuner:
    """Fine-tune models using OpenAI's fine-tuning API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.fine_tuned_model = None

    def prepare_training_file(
        self,
        input_jsonl: str,
        output_file: str,
        validation_split: float = 0.1
    ) -> tuple:
        """
        Prepare and validate training data for OpenAI fine-tuning

        Args:
            input_jsonl: Path to input JSONL file
            output_file: Path for training file
            validation_split: Portion of data for validation

        Returns:
            Tuple of (training_file_path, validation_file_path)
        """
        print(f"Preparing training data from {input_jsonl}...")

        # Read data
        with open(input_jsonl, 'r') as f:
            data = [json.loads(line) for line in f]

        print(f"Total examples: {len(data)}")

        # Split data
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        # Save training file
        train_file = output_file.replace('.jsonl', '_train.jsonl')
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')

        # Save validation file
        val_file = output_file.replace('.jsonl', '_val.jsonl')
        with open(val_file, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')

        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        print(f"Training file: {train_file}")
        print(f"Validation file: {val_file}")

        return train_file, val_file

    def upload_training_file(self, file_path: str) -> str:
        """Upload training file to OpenAI"""
        print(f"\nUploading training file: {file_path}")

        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )

        file_id = response.id
        print(f"File uploaded successfully. File ID: {file_id}")
        return file_id

    def create_fine_tune_job(
        self,
        training_file_id: str,
        model: str = "gpt-3.5-turbo",
        suffix: Optional[str] = None,
        hyperparameters: Optional[Dict] = None
    ) -> str:
        """
        Create a fine-tuning job

        Args:
            training_file_id: ID of uploaded training file
            model: Base model to fine-tune
            suffix: Custom suffix for model name
            hyperparameters: Training hyperparameters

        Returns:
            Fine-tuning job ID
        """
        print(f"\nCreating fine-tuning job...")
        print(f"Base model: {model}")

        if hyperparameters is None:
            hyperparameters = {
                "n_epochs": 3,
                "batch_size": 1,
                "learning_rate_multiplier": 1.0
            }

        kwargs = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": hyperparameters
        }

        if suffix:
            kwargs["suffix"] = suffix

        response = self.client.fine_tuning.jobs.create(**kwargs)

        job_id = response.id
        print(f"Fine-tuning job created. Job ID: {job_id}")
        return job_id

    def monitor_fine_tune_job(self, job_id: str, poll_interval: int = 60):
        """
        Monitor fine-tuning job progress

        Args:
            job_id: Fine-tuning job ID
            poll_interval: Seconds between status checks
        """
        print(f"\nMonitoring fine-tuning job: {job_id}")
        print("This may take several minutes to hours depending on dataset size...")

        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status

            print(f"Status: {status}")

            if status == "succeeded":
                self.fine_tuned_model = job.fine_tuned_model
                print(f"\n✓ Fine-tuning completed successfully!")
                print(f"Fine-tuned model: {self.fine_tuned_model}")
                return self.fine_tuned_model

            elif status == "failed":
                print(f"\n✗ Fine-tuning failed!")
                print(f"Error: {job.error}")
                return None

            elif status in ["running", "pending"]:
                print(f"Job is {status}. Checking again in {poll_interval} seconds...")
                time.sleep(poll_interval)

            else:
                print(f"Unknown status: {status}")
                break

        return None

    def test_fine_tuned_model(
        self,
        model_name: str,
        test_queries: List[str]
    ) -> List[str]:
        """
        Test the fine-tuned model with sample queries

        Args:
            model_name: Fine-tuned model name
            test_queries: List of test queries

        Returns:
            List of model responses
        """
        print(f"\nTesting fine-tuned model: {model_name}")
        print("="*60)

        responses = []

        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}/{len(test_queries)}")
            print(f"Query: {query}")
            print("-"*60)

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful retail customer support agent."},
                    {"role": "user", "content": query}
                ],
                max_tokens=300,
                temperature=0.7
            )

            answer = response.choices[0].message.content
            print(f"Response: {answer}")
            responses.append(answer)

        return responses

    def fine_tune_pipeline(
        self,
        data_file: str,
        model: str = "gpt-3.5-turbo",
        suffix: str = "retail-support",
        test_queries: Optional[List[str]] = None
    ) -> str:
        """
        Complete fine-tuning pipeline

        Args:
            data_file: Path to training data JSONL
            model: Base model to fine-tune
            suffix: Model name suffix
            test_queries: Optional test queries

        Returns:
            Fine-tuned model name
        """
        print("="*70)
        print("OPENAI FINE-TUNING PIPELINE")
        print("="*70)

        # Step 1: Prepare training file
        train_file, val_file = self.prepare_training_file(
            data_file,
            data_file.replace('.jsonl', '_prepared.jsonl')
        )

        # Step 2: Upload training file
        training_file_id = self.upload_training_file(train_file)

        # Step 3: Create fine-tuning job
        job_id = self.create_fine_tune_job(
            training_file_id=training_file_id,
            model=model,
            suffix=suffix
        )

        # Step 4: Monitor job (optional - can be done separately)
        print("\nNote: Fine-tuning job has been created.")
        print(f"Job ID: {job_id}")
        print("\nTo monitor progress, run:")
        print(f"  python -m models.fine_tuning monitor {job_id}")
        print("\nOr monitor in OpenAI dashboard:")
        print("  https://platform.openai.com/finetune")

        return job_id


class HuggingFaceFineTuner:
    """Fine-tune models using Hugging Face with LoRA/PEFT"""

    def __init__(self):
        """Initialize Hugging Face fine-tuner"""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from datasets import load_dataset
            import torch

            self.transformers_available = True
            print("✓ Hugging Face transformers available")
        except ImportError:
            self.transformers_available = False
            print("✗ Hugging Face transformers not available")
            print("Install with: pip install transformers peft datasets torch")

    def prepare_dataset(self, jsonl_file: str):
        """Prepare dataset for Hugging Face training"""
        from datasets import load_dataset

        print(f"Loading dataset from {jsonl_file}...")
        dataset = load_dataset('json', data_files=jsonl_file)

        def format_instruction(example):
            """Format examples for instruction tuning"""
            messages = example['messages']
            formatted = ""

            for msg in messages:
                role = msg['role']
                content = msg['content']

                if role == "system":
                    formatted += f"<|system|>\n{content}\n"
                elif role == "user":
                    formatted += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    formatted += f"<|assistant|>\n{content}\n"

            return {"text": formatted}

        dataset = dataset.map(format_instruction)
        return dataset

    def fine_tune_with_lora(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        dataset_file: str = "../data/processed/finetuning_data.jsonl",
        output_dir: str = "../models/finetuned_model"
    ):
        """
        Fine-tune model using LoRA (Low-Rank Adaptation)

        Args:
            model_name: Base model from Hugging Face
            dataset_file: Training data file
            output_dir: Output directory for fine-tuned model
        """
        if not self.transformers_available:
            print("Error: Transformers not available")
            return

        print("="*70)
        print("HUGGING FACE LORA FINE-TUNING")
        print("="*70)
        print(f"Base model: {model_name}")
        print(f"Dataset: {dataset_file}")
        print(f"Output: {output_dir}")

        # This is a template - actual implementation would require more setup
        print("\nNote: This is a template for Hugging Face fine-tuning.")
        print("For production use, you would need to:")
        print("1. Load the base model and tokenizer")
        print("2. Configure LoRA parameters")
        print("3. Prepare the training dataset")
        print("4. Set up training arguments")
        print("5. Train the model")
        print("6. Save the fine-tuned model")

        return output_dir


def main():
    """Main function for fine-tuning"""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Fine-tune customer support model")
    parser.add_argument('command', choices=['openai', 'huggingface', 'monitor', 'test'])
    parser.add_argument('--data', default='../data/processed/finetuning_data.jsonl')
    parser.add_argument('--model', default='gpt-3.5-turbo')
    parser.add_argument('--job-id', help='Job ID for monitoring')
    parser.add_argument('--model-name', help='Model name for testing')

    args = parser.parse_args()

    if args.command == 'openai':
        # OpenAI fine-tuning
        fine_tuner = OpenAIFineTuner()

        test_queries = [
            "What's the best USB cable under 500 rupees?",
            "Tell me about boAt cables",
            "Which cable supports fast charging?"
        ]

        job_id = fine_tuner.fine_tune_pipeline(
            data_file=args.data,
            model=args.model,
            test_queries=test_queries
        )

    elif args.command == 'monitor':
        # Monitor existing job
        if not args.job_id:
            print("Error: --job-id required for monitoring")
            return

        fine_tuner = OpenAIFineTuner()
        model_name = fine_tuner.monitor_fine_tune_job(args.job_id)

        if model_name:
            print(f"\nFine-tuned model ready: {model_name}")

    elif args.command == 'test':
        # Test fine-tuned model
        if not args.model_name:
            print("Error: --model-name required for testing")
            return

        fine_tuner = OpenAIFineTuner()
        test_queries = [
            "What's the best USB cable?",
            "Tell me about charging cables",
            "Which product has good reviews?"
        ]

        fine_tuner.test_fine_tuned_model(args.model_name, test_queries)

    elif args.command == 'huggingface':
        # Hugging Face fine-tuning
        fine_tuner = HuggingFaceFineTuner()
        fine_tuner.fine_tune_with_lora(dataset_file=args.data)


if __name__ == "__main__":
    main()

"""
RAGAS Evaluation Framework
Evaluates RAG system performance using faithfulness, relevancy, context recall, etc.
"""

import os
import json
from typing import List, Dict, Optional
import pandas as pd

# Fix for Git import error in RAGAS (Windows compatibility)
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RAGASEvaluator:
    """Evaluate RAG system using RAGAS metrics"""

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",  # Using current OpenAI model
        embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize RAGAS evaluator"""
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Available metrics
        self.available_metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_recall': context_recall,
            'context_precision': context_precision,
            'answer_similarity': answer_similarity,
            'answer_correctness': answer_correctness
        }

    def prepare_evaluation_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dataset:
        """
        Prepare dataset for RAGAS evaluation

        Args:
            questions: List of user questions
            answers: List of generated answers
            contexts: List of retrieved contexts (list of strings for each question)
            ground_truths: Optional list of reference answers

        Returns:
            Dataset object for RAGAS
        """
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
        }

        if ground_truths:
            data['ground_truth'] = ground_truths

        return Dataset.from_dict(data)

    def evaluate_rag_system(
        self,
        dataset: Dataset,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate RAG system using specified metrics

        Args:
            dataset: Evaluation dataset
            metrics: List of metric names to use (default: all)

        Returns:
            Evaluation results
        """
        if metrics is None:
            # Use core metrics by default
            metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

        selected_metrics = [self.available_metrics[m] for m in metrics if m in self.available_metrics]

        print("="*70)
        print("RAGAS EVALUATION")
        print("="*70)
        print(f"Dataset size: {len(dataset)}")
        print(f"Metrics: {', '.join(metrics)}")
        print("\nEvaluating... This may take a few minutes.")

        # Run evaluation
        results = evaluate(
            dataset,
            metrics=selected_metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )

        return results

    def create_test_cases(self, rag_system, num_cases: int = 10) -> List[Dict]:
        """
        Create test cases for evaluation

        Args:
            rag_system: RAG system instance
            num_cases: Number of test cases

        Returns:
            List of test case dictionaries
        """
        test_cases = [
            {
                'question': 'What are the best USB cables under 500 rupees?',
                'ground_truth': 'Several good options under 500 rupees include boAt Deuce USB 300 at ₹329, Ambrane Unbreakable at ₹199, and pTron Solero TB301 at ₹149. These cables support fast charging and have good ratings.'
            },
            {
                'question': 'Which cable has the best customer reviews?',
                'ground_truth': 'boAt Deuce USB 300 and Wayona cables have excellent reviews with 4.2 ratings. Customers praise their durability and fast charging capabilities.'
            },
            {
                'question': 'Tell me about charging speed of Ambrane cable',
                'ground_truth': 'Ambrane Unbreakable cable supports 60W/3A fast charging with Quick Charge 3.0 and 480Mbps data sync speed.'
            },
            {
                'question': 'What do customers say about boAt cable quality?',
                'ground_truth': 'Customers appreciate the boAt cable for its nylon braided design, durability with 10000+ bends lifespan, and 3A fast charging capability. It has 94,363 reviews with 4.2 rating.'
            },
            {
                'question': 'Is Wayona cable durable?',
                'ground_truth': 'Yes, Wayona cable features nylon braided design with premium aluminum housing and has passed 10,000+ bending tests. Customers confirm it is durable and sturdy.'
            },
            {
                'question': 'Which cable is best for iPhone?',
                'ground_truth': 'For iPhone, Wayona Nylon Braided Lightning Cable and Portronics Konnect L are good options. They are MFi compatible and support fast charging.'
            },
            {
                'question': 'What is the warranty on boAt cables?',
                'ground_truth': 'boAt cables come with a 2-year warranty from the date of purchase.'
            },
            {
                'question': 'Compare boAt and Ambrane cables',
                'ground_truth': 'boAt Deuce USB 300 is priced at ₹329 with 4.2 rating and 3A charging. Ambrane is cheaper at ₹199 with 4.0 rating and supports higher 60W/3A charging, making it better for laptops too.'
            },
            {
                'question': 'Which cable is longest?',
                'ground_truth': 'Most cables including boAt, Ambrane, and pTron offer 1.5 meters length, which is ideal for comfortable usage.'
            },
            {
                'question': 'What are the key features of pTron cable?',
                'ground_truth': 'pTron Solero TB301 is Made in India, supports 3A fast charging, 480Mbps data sync, has double-braided exterior, passed 10,000 bending tests, and is 1.5 meters long at only ₹149.'
            }
        ]

        return test_cases[:num_cases]

    def run_evaluation_pipeline(
        self,
        rag_system,
        test_cases: Optional[List[Dict]] = None,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete evaluation pipeline

        Args:
            rag_system: RAG system instance
            test_cases: Optional list of test cases (will create if not provided)
            output_file: Optional path to save results

        Returns:
            Results as DataFrame
        """
        print("="*70)
        print("RAG SYSTEM EVALUATION PIPELINE")
        print("="*70)

        # Create test cases if not provided
        if test_cases is None:
            test_cases = self.create_test_cases(rag_system)

        print(f"\nGenerating answers for {len(test_cases)} test cases...")

        questions = []
        ground_truths = []
        answers = []
        contexts_list = []

        for i, case in enumerate(test_cases, 1):
            question = case['question']
            ground_truth = case.get('ground_truth', '')

            print(f"\n[{i}/{len(test_cases)}] Processing: {question}")

            # Retrieve context
            retrieved_docs = rag_system.similarity_search(question, k=3)
            contexts = [doc.page_content for doc in retrieved_docs]

            # Generate answer (simplified - in real scenario use your agent)
            context_str = "\n\n".join(contexts)
            answer = f"Based on the available products: {context_str[:200]}..."

            questions.append(question)
            ground_truths.append(ground_truth)
            answers.append(answer)
            contexts_list.append(contexts)

        # Create dataset
        print("\nCreating evaluation dataset...")
        eval_dataset = self.prepare_evaluation_dataset(
            questions=questions,
            answers=answers,
            contexts=contexts_list,
            ground_truths=ground_truths
        )

        # Evaluate
        print("\nRunning RAGAS evaluation...")
        results = self.evaluate_rag_system(eval_dataset)

        # Convert to DataFrame
        if hasattr(results, 'to_pandas'):
            results_df = results.to_pandas()
        elif isinstance(results, dict):
            # If it's a dict of scores, convert directly
            results_df = pd.DataFrame([results])
        else:
            # Try direct conversion
            try:
                results_df = pd.DataFrame(results)
            except:
                # Fallback: create from dict
                results_df = pd.DataFrame([dict(results)])

        # Display results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)

        print("\nMetric Scores:")
        print(results_df.describe())
        print("\nAll Results:")
        print(results_df)

        # Save results
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")

        return results_df

    def generate_evaluation_report(
        self,
        results_df: pd.DataFrame,
        output_file: str = "../data/evaluation_report.md"
    ):
        """Generate markdown evaluation report"""

        report = """# RAG System Evaluation Report

## Overview
This report presents the evaluation results of the Retail Customer Support RAG system using RAGAS metrics.

## Metrics Explained

### Faithfulness
- Measures how factually accurate the generated answer is compared to the retrieved context
- Score range: 0-1 (higher is better)
- Ideal score: > 0.8

### Answer Relevancy
- Measures how relevant the generated answer is to the question
- Score range: 0-1 (higher is better)
- Ideal score: > 0.8

### Context Precision
- Measures how relevant the retrieved context is to the question
- Score range: 0-1 (higher is better)
- Ideal score: > 0.7

### Context Recall
- Measures how much of the ground truth answer can be found in the retrieved context
- Score range: 0-1 (higher is better)
- Ideal score: > 0.8

## Results Summary

"""

        # Add statistics
        if not results_df.empty:
            report += "### Overall Statistics\n\n"
            report += results_df.describe().to_markdown()
            report += "\n\n"

            report += "### Metric Averages\n\n"
            for col in results_df.columns:
                if col not in ['question', 'answer', 'contexts', 'ground_truth']:
                    avg_score = results_df[col].mean()
                    report += f"- **{col.replace('_', ' ').title()}**: {avg_score:.3f}\n"

        report += "\n## Recommendations\n\n"
        report += "Based on the evaluation results:\n\n"
        report += "1. **Low Faithfulness**: Improve context retrieval or answer generation\n"
        report += "2. **Low Answer Relevancy**: Refine prompt engineering and response formatting\n"
        report += "3. **Low Context Precision**: Optimize retrieval parameters and embeddings\n"
        report += "4. **Low Context Recall**: Increase number of retrieved documents or improve chunking\n"

        # Save report
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"\nEvaluation report saved to: {output_file}")
        return report


def main():
    """Main evaluation function"""
    from dotenv import load_dotenv
    import sys
    import os

    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    load_dotenv()

    # Load RAG system
    print("Loading RAG system...")
    from utils.rag_system import RAGSystem

    rag = RAGSystem(persist_directory="../vectorstore/chroma_db")

    try:
        rag.load_vectorstore()
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Please run data preprocessing and RAG setup first.")
        return

    # Initialize evaluator
    evaluator = RAGASEvaluator()

    # Run evaluation
    results = evaluator.run_evaluation_pipeline(
        rag_system=rag,
        output_file="../data/ragas_results.csv"
    )

    # Generate report
    evaluator.generate_evaluation_report(
        results,
        output_file="../data/evaluation_report.md"
    )

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

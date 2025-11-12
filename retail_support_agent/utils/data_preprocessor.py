"""
Data Preprocessing Module for Amazon Product Dataset
Prepares data for RAG, fine-tuning, and agent usage
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import json


class AmazonDataPreprocessor:
    """Preprocesses Amazon product and review data"""

    def __init__(self, csv_path: str):
        """Initialize with path to CSV file"""
        self.csv_path = csv_path
        self.df = None
        self.processed_products = []
        self.qa_pairs = []

    def load_data(self) -> pd.DataFrame:
        """Load and perform initial cleaning"""
        print("Loading Amazon dataset...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} records")
        print(f"Columns: {list(self.df.columns)}")
        return self.df

    def clean_text(self, text: str) -> str:
        """Clean text data"""
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def extract_product_info(self, row) -> Dict:
        """Extract structured product information"""
        product_info = {
            'product_id': row.get('product_id', ''),
            'product_name': self.clean_text(row.get('product_name', '')),
            'category': row.get('category', ''),
            'discounted_price': row.get('discounted_price', ''),
            'actual_price': row.get('actual_price', ''),
            'discount_percentage': row.get('discount_percentage', ''),
            'rating': row.get('rating', 0),
            'rating_count': row.get('rating_count', 0),
            'about_product': self.clean_text(row.get('about_product', '')),
            'product_link': row.get('product_link', '')
        }
        return product_info

    def extract_reviews(self, row) -> List[Dict]:
        """Extract and structure review information"""
        reviews = []

        # Handle multiple reviews in single row
        user_ids = str(row.get('user_id', '')).split(',')
        user_names = str(row.get('user_name', '')).split(',')
        review_titles = str(row.get('review_title', '')).split(',')
        review_contents = str(row.get('review_content', '')).split(',')

        for i in range(min(len(user_ids), len(review_contents))):
            if i < len(user_names) and i < len(review_titles):
                review = {
                    'user_name': user_names[i].strip() if i < len(user_names) else '',
                    'review_title': self.clean_text(review_titles[i]) if i < len(review_titles) else '',
                    'review_content': self.clean_text(review_contents[i]) if i < len(review_contents) else ''
                }
                if review['review_content']:
                    reviews.append(review)

        return reviews

    def create_rag_documents(self) -> List[Dict]:
        """Create documents optimized for RAG retrieval"""
        print("\nCreating RAG documents...")
        documents = []

        for idx, row in self.df.iterrows():
            product_info = self.extract_product_info(row)
            reviews = self.extract_reviews(row)

            # Create comprehensive product document
            doc_text = f"""
Product: {product_info['product_name']}
Category: {product_info['category']}
Price: {product_info['discounted_price']} (Original: {product_info['actual_price']})
Discount: {product_info['discount_percentage']}
Rating: {product_info['rating']}/5 ({product_info['rating_count']} reviews)

Product Details:
{product_info['about_product']}

Customer Reviews:
"""
            # Add top reviews
            for i, review in enumerate(reviews[:5]):  # Top 5 reviews per product
                doc_text += f"\n- {review['review_title']}: {review['review_content']}"

            document = {
                'id': f"product_{idx}",
                'text': doc_text.strip(),
                'metadata': {
                    'product_id': product_info['product_id'],
                    'product_name': product_info['product_name'],
                    'category': product_info['category'],
                    'rating': product_info['rating'],
                    'price': product_info['discounted_price'],
                    'product_link': product_info['product_link']
                }
            }
            documents.append(document)

        print(f"Created {len(documents)} RAG documents")
        return documents

    def create_finetuning_dataset(self) -> List[Dict]:
        """Create Q&A pairs for fine-tuning"""
        print("\nCreating fine-tuning dataset...")
        training_data = []

        for idx, row in self.df.iterrows():
            product_info = self.extract_product_info(row)
            reviews = self.extract_reviews(row)

            # Generate various types of customer support Q&A pairs

            # 1. Product inquiry
            qa_1 = {
                "messages": [
                    {"role": "system", "content": "You are an expert retail customer support agent helping customers with product inquiries."},
                    {"role": "user", "content": f"Can you tell me about {product_info['product_name']}?"},
                    {"role": "assistant", "content": f"{product_info['product_name']} is available in the {product_info['category']} category. {product_info['about_product'][:200]}... It has a rating of {product_info['rating']}/5 based on {product_info['rating_count']} customer reviews."}
                ]
            }
            training_data.append(qa_1)

            # 2. Price inquiry
            if product_info['discounted_price'] and product_info['actual_price']:
                qa_2 = {
                    "messages": [
                        {"role": "system", "content": "You are an expert retail customer support agent helping customers with pricing inquiries."},
                        {"role": "user", "content": f"What is the price of {product_info['product_name']}?"},
                        {"role": "assistant", "content": f"The current price is {product_info['discounted_price']}, down from {product_info['actual_price']}. That's a {product_info['discount_percentage']} discount!"}
                    ]
                }
                training_data.append(qa_2)

            # 3. Review-based inquiry
            if reviews:
                positive_reviews = [r for r in reviews if 'good' in r['review_content'].lower() or 'great' in r['review_content'].lower()]
                if positive_reviews:
                    qa_3 = {
                        "messages": [
                            {"role": "system", "content": "You are an expert retail customer support agent helping customers understand product reviews."},
                            {"role": "user", "content": f"What do customers say about {product_info['product_name']}?"},
                            {"role": "assistant", "content": f"Customers generally appreciate this product. Here are some highlights: {positive_reviews[0]['review_content'][:150]}... The product has an overall rating of {product_info['rating']}/5."}
                        ]
                    }
                    training_data.append(qa_3)

            # 4. Comparison inquiry
            qa_4 = {
                "messages": [
                    {"role": "system", "content": "You are an expert retail customer support agent helping customers make purchase decisions."},
                    {"role": "user", "content": f"Is {product_info['product_name']} worth buying?"},
                    {"role": "assistant", "content": f"Based on {product_info['rating_count']} customer reviews, this product has a {product_info['rating']}/5 rating. With a {product_info['discount_percentage']} discount, it's currently priced at {product_info['discounted_price']}. Customers particularly mention: {product_info['about_product'][:100]}..."}
                ]
            }
            training_data.append(qa_4)

        print(f"Created {len(training_data)} training examples")
        return training_data

    def save_processed_data(self, rag_documents: List[Dict], training_data: List[Dict], output_dir: str):
        """Save processed data to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save RAG documents
        with open(f"{output_dir}/rag_documents.json", 'w') as f:
            json.dump(rag_documents, f, indent=2)
        print(f"\nSaved RAG documents to {output_dir}/rag_documents.json")

        # Save fine-tuning data in JSONL format
        with open(f"{output_dir}/finetuning_data.jsonl", 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved fine-tuning data to {output_dir}/finetuning_data.jsonl")

        # Save processed products as CSV for ML analysis
        processed_df = self.df.copy()
        processed_df.to_csv(f"{output_dir}/processed_products.csv", index=False)
        print(f"Saved processed products to {output_dir}/processed_products.csv")

        # Save summary statistics
        stats = {
            'total_products': len(rag_documents),
            'total_training_examples': len(training_data),
            'categories': self.df['category'].nunique() if 'category' in self.df.columns else 0,
            'avg_rating': float(self.df['rating'].mean()) if 'rating' in self.df.columns else 0
        }
        with open(f"{output_dir}/dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved dataset statistics to {output_dir}/dataset_stats.json")

        return stats


def main():
    """Main preprocessing pipeline"""
    preprocessor = AmazonDataPreprocessor('./data/amazon.csv')

    # Load data
    df = preprocessor.load_data()

    # Create RAG documents
    rag_documents = preprocessor.create_rag_documents()

    # Create fine-tuning dataset
    training_data = preprocessor.create_finetuning_dataset()

    # Save processed data
    stats = preprocessor.save_processed_data(rag_documents, training_data, './data/processed')

    print("\n" + "="*50)
    print("Preprocessing Complete!")
    print("="*50)
    print(f"Total Products: {stats['total_products']}")
    print(f"Training Examples: {stats['total_training_examples']}")
    print(f"Categories: {stats['categories']}")
    print(f"Average Rating: {stats['avg_rating']:.2f}")


if __name__ == "__main__":
    main()

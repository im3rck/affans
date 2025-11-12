"""
Hybrid ML + GenAI System
Combines traditional ML (clustering, classification) with LLM interpretation
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import json


class HybridMLSystem:
    """
    Hybrid ML + GenAI System:
    1. Clustering - Groups similar products
    2. Classification - Predicts product quality
    3. Price Analysis - Identifies value segments
    4. LLM Interpretation - Translates ML findings into natural language
    """

    def __init__(
        self,
        data_path: str = "./data/processed/processed_products.csv",
        llm_model: str = "gpt-3.5-turbo"
    ):
        """Initialize Hybrid ML System"""
        print("🤖 Initializing Hybrid ML + GenAI System...")

        # Load data
        print(f"   Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        print(f"   Loaded {len(self.df)} products")

        # Initialize LLM for interpretation
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)

        # Prepare features
        self._prepare_features()

        # Initialize models
        self.clusterer = None
        self.classifier = None
        self.scaler = StandardScaler()

        print("✅ Hybrid ML System initialized!\n")

    def _prepare_features(self):
        """Extract numerical features for ML"""
        print("   Preparing features...")

        # Extract price numbers
        self.df['price_numeric'] = self.df['discounted_price'].apply(
            lambda x: self._extract_price(str(x))
        )

        # Clean ratings
        self.df['rating_numeric'] = pd.to_numeric(
            self.df['rating'],
            errors='coerce'
        ).fillna(0)

        # Rating count
        self.df['rating_count_numeric'] = self.df['rating_count'].apply(
            lambda x: self._extract_number(str(x))
        )

        # Create feature matrix
        self.feature_cols = ['price_numeric', 'rating_numeric', 'rating_count_numeric']
        self.features = self.df[self.feature_cols].fillna(0)

        print(f"   Features prepared: {self.feature_cols}")

    @staticmethod
    def _extract_price(price_str: str) -> float:
        """Extract numeric price from string"""
        try:
            # Remove currency symbols and commas
            price_clean = ''.join(filter(str.isdigit, price_str.split(',')[0]))
            return float(price_clean) if price_clean else 0
        except:
            return 0

    @staticmethod
    def _extract_number(num_str: str) -> float:
        """Extract numeric value from string"""
        try:
            num_clean = ''.join(filter(str.isdigit, num_str.replace(',', '')))
            return float(num_clean) if num_clean else 0
        except:
            return 0

    def cluster_products(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Product Clustering using K-means
        Groups similar products based on price, rating, and popularity
        """
        print(f"\n📊 Clustering products into {n_clusters} segments...")

        # Scale features
        features_scaled = self.scaler.fit_transform(self.features)

        # Fit K-means
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.clusterer.fit_predict(features_scaled)

        print("   ✅ Clustering complete!")

        # Analyze clusters
        cluster_analysis = []
        for i in range(n_clusters):
            cluster_data = self.df[self.df['cluster'] == i]

            analysis = {
                'cluster_id': i,
                'size': len(cluster_data),
                'avg_price': cluster_data['price_numeric'].mean(),
                'avg_rating': cluster_data['rating_numeric'].mean(),
                'avg_reviews': cluster_data['rating_count_numeric'].mean(),
                'price_range': (
                    cluster_data['price_numeric'].min(),
                    cluster_data['price_numeric'].max()
                ),
                'sample_products': cluster_data['product_name'].head(3).tolist()
            }
            cluster_analysis.append(analysis)

            print(f"\n   Cluster {i}:")
            print(f"      Size: {analysis['size']} products")
            print(f"      Avg Price: ₹{analysis['avg_price']:.0f}")
            print(f"      Avg Rating: {analysis['avg_rating']:.2f}/5")

        return {
            'n_clusters': n_clusters,
            'clusters': cluster_analysis,
            'total_products': len(self.df)
        }

    def classify_quality(self) -> Dict[str, Any]:
        """
        Quality Classification using Random Forest
        Predicts if a product is 'high quality' based on features
        """
        print("\n🎯 Training quality classifier...")

        # Create quality labels (rating >= 4.0 = high quality)
        self.df['high_quality'] = (self.df['rating_numeric'] >= 4.0).astype(int)

        # Prepare data
        X = self.features.values
        y = self.df['high_quality'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.classifier.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)

        print(f"   ✅ Classifier trained!")
        print(f"      Training accuracy: {train_score:.2%}")
        print(f"      Test accuracy: {test_score:.2%}")

        # Feature importance
        importances = self.classifier.feature_importances_
        feature_importance = dict(zip(self.feature_cols, importances))

        # Quality distribution
        quality_dist = self.df['high_quality'].value_counts()

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance,
            'quality_distribution': {
                'high_quality': int(quality_dist.get(1, 0)),
                'low_quality': int(quality_dist.get(0, 0))
            }
        }

    def analyze_price_segments(self) -> Dict[str, Any]:
        """
        Price Segmentation Analysis
        Identifies budget, mid-range, and premium segments
        """
        print("\n💰 Analyzing price segments...")

        prices = self.df['price_numeric']
        prices = prices[prices > 0]  # Remove zero prices

        # Define segments using percentiles
        q25 = prices.quantile(0.25)
        q75 = prices.quantile(0.75)

        # Categorize
        self.df['price_segment'] = 'Mid-range'
        self.df.loc[self.df['price_numeric'] < q25, 'price_segment'] = 'Budget'
        self.df.loc[self.df['price_numeric'] > q75, 'price_segment'] = 'Premium'

        # Analyze segments
        segments = {}
        for segment in ['Budget', 'Mid-range', 'Premium']:
            seg_data = self.df[self.df['price_segment'] == segment]

            segments[segment] = {
                'count': len(seg_data),
                'percentage': len(seg_data) / len(self.df) * 100,
                'price_range': (
                    seg_data['price_numeric'].min(),
                    seg_data['price_numeric'].max()
                ),
                'avg_price': seg_data['price_numeric'].mean(),
                'avg_rating': seg_data['rating_numeric'].mean(),
                'top_products': seg_data.nlargest(3, 'rating_numeric')['product_name'].tolist()
            }

            print(f"\n   {segment}:")
            print(f"      Count: {segments[segment]['count']} ({segments[segment]['percentage']:.1f}%)")
            print(f"      Price range: ₹{segments[segment]['price_range'][0]:.0f} - ₹{segments[segment]['price_range'][1]:.0f}")
            print(f"      Avg rating: {segments[segment]['avg_rating']:.2f}/5")

        return {
            'segments': segments,
            'thresholds': {
                'budget_max': q25,
                'premium_min': q75
            }
        }

    def sentiment_analysis(self) -> Dict[str, Any]:
        """
        Simple sentiment analysis based on ratings
        """
        print("\n😊 Analyzing sentiment...")

        # Sentiment categories
        self.df['sentiment'] = 'Neutral'
        self.df.loc[self.df['rating_numeric'] >= 4.0, 'sentiment'] = 'Positive'
        self.df.loc[self.df['rating_numeric'] < 3.0, 'sentiment'] = 'Negative'

        sentiment_dist = self.df['sentiment'].value_counts()

        result = {
            'positive': int(sentiment_dist.get('Positive', 0)),
            'neutral': int(sentiment_dist.get('Neutral', 0)),
            'negative': int(sentiment_dist.get('Negative', 0)),
            'positive_percentage': sentiment_dist.get('Positive', 0) / len(self.df) * 100,
            'avg_rating': self.df['rating_numeric'].mean()
        }

        print(f"   Positive: {result['positive']} ({result['positive_percentage']:.1f}%)")
        print(f"   Neutral: {result['neutral']}")
        print(f"   Negative: {result['negative']}")

        return result

    def interpret_with_llm(self, analysis_type: str, data: Dict[str, Any]) -> str:
        """
        LLM Interpretation: Translates ML findings into natural language
        This is where GenAI adds value by explaining technical results
        """
        print(f"\n🧠 LLM interpreting {analysis_type} results...")

        prompts = {
            'clustering': self._create_clustering_prompt(data),
            'classification': self._create_classification_prompt(data),
            'price_segments': self._create_price_prompt(data),
            'sentiment': self._create_sentiment_prompt(data)
        }

        prompt = prompts.get(analysis_type, "Summarize this data: " + str(data))

        try:
            response = self.llm.invoke(prompt)
            interpretation = response.content.strip()
            print("   ✅ Interpretation generated!")
            return interpretation
        except Exception as e:
            print(f"   ⚠️ LLM interpretation failed: {e}")
            return f"Analysis completed. See raw data for details."

    def _create_clustering_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for clustering interpretation"""
        clusters_info = "\n".join([
            f"Cluster {c['cluster_id']}: {c['size']} products, "
            f"avg price ₹{c['avg_price']:.0f}, "
            f"avg rating {c['avg_rating']:.2f}/5, "
            f"example: {c['sample_products'][0] if c['sample_products'] else 'N/A'}"
            for c in data['clusters']
        ])

        return f"""Analyze these product clusters and provide insights:

Total Products: {data['total_products']}
Number of Clusters: {data['n_clusters']}

Clusters:
{clusters_info}

Provide a 2-3 sentence business insight about:
1. What each cluster represents (budget/premium/mid-range)
2. Which clusters offer the best value
3. Recommendations for different customer segments

Keep it concise and actionable."""

    def _create_classification_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for classification interpretation"""
        return f"""Analyze this product quality classification model:

Model Performance:
- Training accuracy: {data['train_accuracy']:.1%}
- Test accuracy: {data['test_accuracy']:.1%}

Feature Importance:
{json.dumps(data['feature_importance'], indent=2)}

Quality Distribution:
- High quality products: {data['quality_distribution']['high_quality']}
- Lower quality products: {data['quality_distribution']['low_quality']}

Provide 2-3 sentences explaining:
1. How well the model performs
2. Which features matter most for quality
3. Business implications

Keep it concise and non-technical."""

    def _create_price_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for price analysis interpretation"""
        segments_info = "\n".join([
            f"{name}: {info['count']} products ({info['percentage']:.1f}%), "
            f"₹{info['avg_price']:.0f} avg, "
            f"{info['avg_rating']:.2f}/5 rating"
            for name, info in data['segments'].items()
        ])

        return f"""Analyze these price segments:

{segments_info}

Thresholds:
- Budget: Up to ₹{data['thresholds']['budget_max']:.0f}
- Premium: Above ₹{data['thresholds']['premium_min']:.0f}

Provide 2-3 sentences about:
1. Value for money in each segment
2. Which segment offers best quality-to-price ratio
3. Recommendations for buyers

Keep it actionable and concise."""

    def _create_sentiment_prompt(self, data: Dict[str, Any]) -> str:
        """Create prompt for sentiment interpretation"""
        return f"""Analyze customer sentiment from ratings:

Distribution:
- Positive (4+ stars): {data['positive']} products ({data['positive_percentage']:.1f}%)
- Neutral (3-4 stars): {data['neutral']} products
- Negative (<3 stars): {data['negative']} products

Average Rating: {data['avg_rating']:.2f}/5

Provide 2-3 sentences about:
1. Overall customer satisfaction
2. What this means for product quality
3. Key takeaways

Keep it brief and business-focused."""

    def complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete hybrid ML + GenAI analysis
        Returns both ML results and LLM interpretations
        """
        print("=" * 80)
        print("🚀 COMPLETE HYBRID ML + GENAI ANALYSIS")
        print("=" * 80)

        results = {}

        # 1. Clustering
        print("\n" + "-" * 80)
        clustering_data = self.cluster_products(n_clusters=4)
        clustering_insight = self.interpret_with_llm('clustering', clustering_data)
        results['clustering'] = {
            'data': clustering_data,
            'insight': clustering_insight
        }

        # 2. Classification
        print("\n" + "-" * 80)
        classification_data = self.classify_quality()
        classification_insight = self.interpret_with_llm('classification', classification_data)
        results['classification'] = {
            'data': classification_data,
            'insight': classification_insight
        }

        # 3. Price Segmentation
        print("\n" + "-" * 80)
        price_data = self.analyze_price_segments()
        price_insight = self.interpret_with_llm('price_segments', price_data)
        results['price_segments'] = {
            'data': price_data,
            'insight': price_insight
        }

        # 4. Sentiment
        print("\n" + "-" * 80)
        sentiment_data = self.sentiment_analysis()
        sentiment_insight = self.interpret_with_llm('sentiment', sentiment_data)
        results['sentiment'] = {
            'data': sentiment_data,
            'insight': sentiment_insight
        }

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80 + "\n")

        return results


def main():
    """Demo the Hybrid ML system"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 80)
    print("HYBRID ML + GENAI SYSTEM DEMO")
    print("=" * 80 + "\n")

    # Initialize
    ml_system = HybridMLSystem()

    # Run complete analysis
    results = ml_system.complete_analysis()

    # Display insights
    print("\n" + "=" * 80)
    print("📊 AI-GENERATED INSIGHTS")
    print("=" * 80)

    print("\n🔹 CLUSTERING INSIGHTS:")
    print(results['clustering']['insight'])

    print("\n🔹 QUALITY CLASSIFICATION INSIGHTS:")
    print(results['classification']['insight'])

    print("\n🔹 PRICE ANALYSIS INSIGHTS:")
    print(results['price_segments']['insight'])

    print("\n🔹 SENTIMENT INSIGHTS:")
    print(results['sentiment']['insight'])


if __name__ == "__main__":
    main()

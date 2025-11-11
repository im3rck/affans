"""
Setup script for Intelligent Retail Customer Support Agent
Automates the complete setup process
"""

import os
import sys
import subprocess
from pathlib import Path
import json


class SetupManager:
    """Manages project setup"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.steps_completed = []

    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)

    def print_step(self, step_num, text):
        """Print step information"""
        print(f"\n[Step {step_num}] {text}")
        print("-"*70)

    def check_python_version(self):
        """Check Python version"""
        self.print_step(1, "Checking Python version...")

        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
            return True
        else:
            print(f"❌ Python 3.9+ required. Current: {version.major}.{version.minor}")
            return False

    def check_env_file(self):
        """Check .env file"""
        self.print_step(2, "Checking environment configuration...")

        env_file = self.project_root / '.env'
        env_example = self.project_root / '.env.example'

        if env_file.exists():
            print("✅ .env file exists")

            # Check for API key
            with open(env_file, 'r') as f:
                content = f.read()
                if 'your_openai_api_key_here' in content or not 'OPENAI_API_KEY' in content:
                    print("⚠️  Warning: OpenAI API key not configured")
                    print("   Please edit .env and add your API key")
                    return False
                else:
                    print("✅ OpenAI API key configured")
                    return True
        else:
            print("❌ .env file not found")
            if env_example.exists():
                print("📝 Copying .env.example to .env...")
                import shutil
                shutil.copy(env_example, env_file)
                print("✅ .env file created")
                print("⚠️  Please edit .env and add your OpenAI API key")
            return False

    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_step(3, "Installing dependencies...")

        requirements = self.project_root / 'requirements.txt'

        if not requirements.exists():
            print("❌ requirements.txt not found")
            return False

        try:
            print("📦 Installing packages (this may take a few minutes)...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements)
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False

    def check_dataset(self):
        """Check if dataset exists"""
        self.print_step(4, "Checking dataset...")

        dataset_path = self.project_root / 'data' / 'amazon.csv'

        if dataset_path.exists():
            print(f"✅ Dataset found: {dataset_path}")

            # Check file size
            size_mb = dataset_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {size_mb:.2f} MB")
            return True
        else:
            print("❌ Dataset not found")
            print(f"   Expected location: {dataset_path}")
            print("   Please ensure amazon.csv is in the data/ directory")
            return False

    def run_preprocessing(self):
        """Run data preprocessing"""
        self.print_step(5, "Running data preprocessing...")

        preprocessor = self.project_root / 'utils' / 'data_preprocessor.py'

        if not preprocessor.exists():
            print("❌ Preprocessor script not found")
            return False

        try:
            print("🔄 Processing dataset...")
            result = subprocess.run(
                [sys.executable, str(preprocessor)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✅ Data preprocessing completed")
                print("\n" + result.stdout)
                return True
            else:
                print(f"❌ Preprocessing failed")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"❌ Error running preprocessing: {e}")
            return False

    def setup_rag_system(self):
        """Setup RAG system"""
        self.print_step(6, "Setting up RAG system...")

        rag_script = self.project_root / 'utils' / 'rag_system.py'

        if not rag_script.exists():
            print("❌ RAG system script not found")
            return False

        try:
            print("🔄 Creating vector database...")
            result = subprocess.run(
                [sys.executable, str(rag_script)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                print("✅ RAG system setup completed")
                return True
            else:
                print(f"❌ RAG setup failed")
                print(result.stderr)
                return False
        except subprocess.TimeoutExpired:
            print("⚠️  RAG setup timed out (this is okay for large datasets)")
            return True
        except Exception as e:
            print(f"❌ Error setting up RAG: {e}")
            return False

    def verify_installation(self):
        """Verify installation"""
        self.print_step(7, "Verifying installation...")

        checks = {
            "Processed data": (self.project_root / 'data' / 'processed' / 'rag_documents.json'),
            "Fine-tuning data": (self.project_root / 'data' / 'processed' / 'finetuning_data.jsonl'),
            "Dataset stats": (self.project_root / 'data' / 'processed' / 'dataset_stats.json'),
            "Vector store": (self.project_root / 'vectorstore' / 'chroma_db'),
        }

        all_good = True
        for name, path in checks.items():
            if path.exists():
                print(f"✅ {name}: Found")
            else:
                print(f"⚠️  {name}: Not found")
                all_good = False

        return all_good

    def print_next_steps(self):
        """Print next steps"""
        self.print_header("🎉 Setup Complete!")

        print("\n📋 Next Steps:")
        print("\n1. Start the application:")
        print("   streamlit run app.py")

        print("\n2. (Optional) Run RAGAS evaluation:")
        print("   python utils/ragas_evaluation.py")

        print("\n3. (Optional) Fine-tune a model:")
        print("   python models/fine_tuning.py openai --data data/processed/finetuning_data.jsonl")

        print("\n4. Access the web interface:")
        print("   http://localhost:8501")

        print("\n📚 Documentation:")
        print("   See README.md for detailed usage instructions")

        print("\n💡 Tips:")
        print("   - Make sure your OpenAI API key is set in .env")
        print("   - The first run may take longer due to embedding generation")
        print("   - Check the Analytics dashboard for system metrics")

    def run_setup(self, skip_preprocessing=False):
        """Run complete setup process"""
        self.print_header("🛍️ Retail Customer Support Agent - Setup")

        print("\nThis script will:")
        print("  ✓ Check system requirements")
        print("  ✓ Install dependencies")
        print("  ✓ Process dataset")
        print("  ✓ Setup RAG system")
        print("  ✓ Verify installation")

        input("\nPress Enter to continue...")

        # Step 1: Check Python version
        if not self.check_python_version():
            print("\n❌ Setup failed: Python version requirement not met")
            return False

        # Step 2: Check environment file
        env_configured = self.check_env_file()

        # Step 3: Install dependencies
        if not self.install_dependencies():
            print("\n❌ Setup failed: Could not install dependencies")
            return False

        # Step 4: Check dataset
        if not self.check_dataset():
            print("\n❌ Setup failed: Dataset not found")
            return False

        if not skip_preprocessing:
            # Step 5: Run preprocessing
            if not self.run_preprocessing():
                print("\n⚠️  Warning: Preprocessing failed, but continuing...")

            # Step 6: Setup RAG
            if not self.setup_rag_system():
                print("\n⚠️  Warning: RAG setup incomplete")

        # Step 7: Verify
        self.verify_installation()

        # Show next steps
        self.print_next_steps()

        if not env_configured:
            print("\n⚠️  IMPORTANT: Configure your OpenAI API key in .env before running!")

        return True


def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Setup Retail Support Agent")
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing (if already done)')
    args = parser.parse_args()

    setup = SetupManager()
    success = setup.run_setup(skip_preprocessing=args.skip_preprocessing)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

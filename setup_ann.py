"""
Setup script to train the ANN model for fertilizer recommendation.
Run this script to create the necessary model files.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def train_model():
    """Train the ANN model"""
    print("Training ANN model...")
    try:
        subprocess.check_call([sys.executable, "train_ann_model.py"])
        print("✅ ANN model trained successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training model: {e}")
        return False
    return True

def check_files():
    """Check if required files exist"""
    required_files = ['f2.csv', 'classifier.pkl', 'fertilizer.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found!")
    return True

def main():
    print("🚀 Setting up ANN Fertilizer Recommendation System")
    print("=" * 50)
    
    # Check if required files exist
    if not check_files():
        print("\n❌ Setup failed: Missing required files")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install requirements")
        return
    
    # Train the model
    if not train_model():
        print("\n❌ Setup failed: Could not train model")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("You can now run the Flask application with: python main.py")
    print("The ANN model will be available at the /ANN route")

if __name__ == "__main__":
    main()
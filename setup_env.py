#!/usr/bin/env python3
"""
Environment setup script for Stock ML Project.
This script helps you create a .env file from the template.
"""

import os
import shutil
from pathlib import Path


def setup_environment():
    """Set up environment file from template."""
    project_root = Path(__file__).parent
    template_file = project_root / "env_template.txt"
    env_file = project_root / ".env"
    
    print("Stock ML Project - Environment Setup")
    print("=" * 40)
    
    # Check if .env already exists
    if env_file.exists():
        response = input(".env file already exists. Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Copy template to .env
    try:
        shutil.copy2(template_file, env_file)
        print(f"✓ Created .env file from template")
        
        # Get API key from user
        print("\n" + "=" * 40)
        print("API KEY SETUP")
        print("=" * 40)
        print("You need a Twelve Data API key to fetch stock data.")
        print("Get your free API key from: https://twelvedata.com/")
        print()
        
        api_key = input("Enter your Twelve Data API key: ").strip()
        
        if api_key:
            # Update the .env file with the actual API key
            with open(env_file, 'r') as f:
                content = f.read()
            
            content = content.replace("your_twelvedata_api_key_here", api_key)
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✓ API key saved to .env file")
        else:
            print("⚠ No API key provided. You'll need to edit .env manually.")
        
        print("\n" + "=" * 40)
        print("SETUP COMPLETE")
        print("=" * 40)
        print("Your .env file has been created with the following structure:")
        print("- TWELVEDATA_API_KEY: Required for data fetching")
        print("- Various optional configuration variables")
        print()
        print("You can now run the stock prediction pipeline:")
        print("python -m src.main --ticker AAPL --tf 15m --bars 5000")
        print()
        print("To modify settings, edit the .env file directly.")
        
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return


if __name__ == "__main__":
    setup_environment()

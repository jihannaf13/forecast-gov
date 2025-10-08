"""
Script to run the Rice Production Monitoring Dashboard
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages if not already installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🌾 Starting Rice Production Monitoring Dashboard...")
    print("📊 Dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "monitoring_system.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])

if __name__ == "__main__":
    install_requirements()
    run_dashboard()
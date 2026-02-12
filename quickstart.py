"""
Interactive Quickstart Launcher for TMNF DQN Training
Checks dependencies, validates files, tests game connection
"""

import os
import sys
import socket
import subprocess


class EnvironmentChecker:
    """Check system and dependencies"""
    
    @staticmethod
    def check_python_version():
        """Verify Python 3.8+"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            return False, f"Python {version.major}.{version.minor} (need 3.8+)"
        return True, f"Python {version.major}.{version.minor}"
    
    @staticmethod
    def check_dependencies():
        """Check if all required packages are installed"""
        packages = {
            'gymnasium': 'gymnasium',
            'torch': 'torch',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib'
        }
        
        missing = []
        for name, import_name in packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(name)
        
        return missing
    
    @staticmethod
    def check_files():
        """Check if core files exist"""
        required_files = [
            'tmnf_env.py',
            'train_dqn.py',
            'utils.py'
        ]
        
        missing = []
        for f in required_files:
            if not os.path.exists(f):
                missing.append(f)
        
        return missing
    
    @staticmethod
    def test_game_connection():
        """Test connection to game socket"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex(('127.0.0.1', 5555))
            sock.close()
            return result == 0
        except:
            return False


class InteractiveLauncher:
    """Interactive menu for launching training"""
    
    @staticmethod
    def show_banner():
        """Display banner"""
        print("\n" + "="*70)
        print("  TMNF DQN TRAINING SYSTEM - INTERACTIVE LAUNCHER")
        print("="*70)
    
    @staticmethod
    def check_setup():
        """Run all checks"""
        print("\n📋 SYSTEM CHECK")
        print("-" * 70)
        
        # Python version
        ok, msg = EnvironmentChecker.check_python_version()
        status = "✓" if ok else "✗"
        print(f"{status} Python: {msg}")
        
        # Dependencies
        missing_deps = EnvironmentChecker.check_dependencies()
        if missing_deps:
            print(f"✗ Missing packages: {', '.join(missing_deps)}")
            print(f"  Install with: pip install {' '.join(missing_deps)}")
            return False
        else:
            print("✓ All dependencies installed")
        
        # Files
        missing_files = EnvironmentChecker.check_files()
        if missing_files:
            print(f"✗ Missing files: {', '.join(missing_files)}")
            return False
        else:
            print("✓ All core files present")
        
        # Game connection
        print("\n🎮 GAME CHECK")
        print("-" * 70)
        print("Testing connection to TMInterface...")
        
        if EnvironmentChecker.test_game_connection():
            print("✓ Game is running and reachable (127.0.0.1:5555)")
            return True
        else:
            print("✗ Cannot reach game!")
            print("\nMake sure:")
            print("  1. TMInterface is running")
            print("  2. Plugin is loaded")
            print("  3. You're in an active race (not menu)")
            print("  4. Plugin port is 5555")
            return False
    
    @staticmethod
    def show_menu():
        """Display main menu"""
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("1. Start training (new or resume)")
        print("2. Evaluate trained model")
        print("3. Plot training curves")
        print("4. Test environment")
        print("5. View documentation")
        print("6. Exit")
        print("="*70)
        
        choice = input("\nSelect option (1-6): ").strip()
        return choice
    
    @staticmethod
    def start_training():
        """Start training"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        print("\n⚙️ Configuration:")
        print("-" * 70)
        print("State size: 6 (x, z, speed, checkpoint, dx, dz)")
        print("Action size: 5 (noop, left, right, accel, brake)")
        print("Network: 128-128-64 hidden layers")
        print("Algorithm: Double DQN with Experience Replay")
        print("Learning rate: 1e-4")
        print("Gamma: 0.99")
        print("Buffer capacity: 10,000")
        print("Save checkpoint every: 50 episodes")
        
        print("\n🚀 Launching training...")
        print("-" * 70)
        
        try:
            subprocess.run([sys.executable, "train_dqn.py"], check=False)
        except KeyboardInterrupt:
            print("\n✓ Training stopped by user")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    @staticmethod
    def evaluate_model():
        """Evaluate trained model"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        print("\nLaunching evaluation tool...")
        print("-" * 70)
        
        try:
            subprocess.run([sys.executable, "utils.py"], check=False)
        except Exception as e:
            print(f"✗ Error: {e}")
    
    @staticmethod
    def show_docs():
        """Show documentation links"""
        print("\n" + "="*70)
        print("DOCUMENTATION")
        print("="*70)
        
        docs = {
            '1': ('QUICK_ANSWER.txt', 'Quick answer to your questions'),
            '2': ('ANSWER_YOUR_QUESTION.md', 'Full technical explanation'),
            '3': ('IMPLEMENTATION_CHECKLIST.md', 'Code implementation guide'),
            '4': ('RESET_AND_COMMUNICATION_GUIDE.md', 'Communication & reset details'),
            '5': ('README.md', 'Complete technical documentation'),
        }
        
        print("\nAvailable documentation:")
        for key, (file, desc) in docs.items():
            print(f"  {key}. {desc}")
            if os.path.exists(file):
                print(f"     ✓ {file}")
            else:
                print(f"     ✗ {file} (not found)")
        
        print("\n6. Back to menu")
        
        choice = input("\nSelect (1-6): ").strip()
        
        if choice in docs:
            filename = docs[choice][0]
            if os.path.exists(filename):
                print(f"\n✓ Opening {filename}...")
                # Try to open with default viewer
                if sys.platform == 'win32':
                    os.startfile(filename)
                elif sys.platform == 'darwin':
                    subprocess.run(['open', filename])
                else:
                    subprocess.run(['xdg-open', filename])
            else:
                print(f"✗ File not found: {filename}")
    
    @staticmethod
    def run(self):
        """Main launcher loop"""
        InteractiveLauncher.show_banner()
        
        if not InteractiveLauncher.check_setup():
            print("\n⚠️ Setup check failed!")
            print("Fix the issues above before starting training.")
            input("\nPress Enter to exit...")
            return
        
        print("\n✓ All checks passed! Ready to train.")
        
        while True:
            choice = InteractiveLauncher.show_menu()
            
            if choice == "1":
                InteractiveLauncher.start_training()
            
            elif choice == "2":
                InteractiveLauncher.evaluate_model()
            
            elif choice == "3":
                print("\n" + "="*70)
                print("PLOTTING TRAINING CURVES")
                print("="*70)
                try:
                    from utils import TrainingVisualizer
                    TrainingVisualizer.plot_training_curves()
                except Exception as e:
                    print(f"✗ Error: {e}")
            
            elif choice == "4":
                print("\n" + "="*70)
                print("TESTING ENVIRONMENT")
                print("="*70)
                try:
                    from utils import test_environment
                    test_environment()
                except Exception as e:
                    print(f"✗ Error: {e}")
            
            elif choice == "5":
                InteractiveLauncher.show_docs()
            
            elif choice == "6":
                print("\n✓ Goodbye!")
                break
            
            else:
                print("✗ Invalid choice")


def main():
    """Entry point"""
    launcher = InteractiveLauncher()
    launcher.run(launcher)


if __name__ == "__main__":
    main()

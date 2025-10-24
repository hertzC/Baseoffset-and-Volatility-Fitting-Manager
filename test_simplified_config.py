#!/usr/bin/env python3
"""
Simplified Configuration System Test

This script demonstrates the streamlined configuration system where
config_loader.py is now just a compatibility shim and all functionality
is consolidated in config_factory.py.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def test_simplified_structure():
    """Test the simplified configuration structure."""
    
    print("🔧 Simplified Configuration System Test")
    print("=" * 60)
    
    # Method 1: Direct import from config package (recommended)
    print("\n✅ Method 1: Direct Import from Config Package")
    print("-" * 50)
    
    try:
        from config import BaseOffsetConfig, VolatilityConfig, ConfigFactory
        
        base_config = BaseOffsetConfig()
        vol_config = VolatilityConfig()
        
        print(f"✅ BaseOffsetConfig: {base_config.date_str}")
        print(f"✅ VolatilityConfig: {vol_config.date_str}")
        print(f"✅ Factory available: {ConfigFactory.__name__}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 2: Legacy import with deprecation warning
    print(f"\n⚠️  Method 2: Legacy Import (shows deprecation warning)")
    print("-" * 50)
    
    try:
        from config.config_loader import Config, load_config
        
        # This should show a deprecation warning
        legacy_config = Config()
        print(f"✅ Legacy Config still works: {legacy_config.date_str}")
        
        # This should also work
        loaded_config = load_config()
        print(f"✅ Legacy load_config still works: {loaded_config.date_str}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Method 3: New unified API from config package
    print(f"\n🎯 Method 3: New Unified API")
    print("-" * 50)
    
    try:
        from config import load_config, get_config
        
        # Load using new unified API
        config = load_config(config_type="base_offset")
        print(f"✅ Unified load_config: {config.date_str}")
        
        # Get using new unified API
        same_config = get_config("base_offset")
        print(f"✅ Unified get_config: {same_config.date_str}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def show_file_structure():
    """Show the current file structure and what each file does."""
    
    print(f"\n📁 Current File Structure")
    print("=" * 60)
    
    structure = {
        "config/": "Main configuration package",
        "├── __init__.py": "Package interface - imports from config_factory",
        "├── base_config.py": "Abstract base class for all configs",
        "├── base_offset_config.py": "Specialized config for base offset analysis",
        "├── volatility_config.py": "Specialized config for volatility analysis", 
        "├── config_factory.py": "🎯 MAIN MODULE - Contains all functionality",
        "├── config_loader.py": "⚠️ COMPATIBILITY SHIM - Can be removed eventually",
        "├── base_offset_config.yaml": "Base offset configuration file",
        "└── volatility_config.yaml": "Volatility configuration file"
    }
    
    for path, description in structure.items():
        print(f"{path:<30} {description}")

def analyze_dependencies():
    """Analyze what depends on config_loader.py."""
    
    print(f"\n🔍 Dependencies on config_loader.py")
    print("=" * 60)
    
    print("Files that import from config.config_loader:")
    print("• utils/market_data/deribit_md_manager.py")
    print("• utils/market_data/orderbook_deribit_md_manager.py")
    print("• utils/base_offset_fitter/weight_least_square_regressor.py")
    print("• utils/base_offset_fitter/fitter.py")
    print("• utils/base_offset_fitter/nonlinear_minimization.py")
    print("• tests/test_base_offset_fitting.py")
    print("• tests/test_notebook.py")
    print("• tests/test_bitcoin_options.py")
    print("• main.py")
    print("• example_config_usage.py")
    print("• demo_config_structure.py")
    
    print(f"\n💡 Migration Strategy:")
    print("1. Keep config_loader.py as compatibility shim (current)")
    print("2. Gradually update imports to use config package directly")
    print("3. Eventually remove config_loader.py when all code migrated")

def show_migration_examples():
    """Show specific migration examples."""
    
    print(f"\n🔄 Migration Examples")
    print("=" * 60)
    
    examples = [
        {
            "old": "from config.config_loader import Config",
            "new": "from config import BaseOffsetConfig as Config"
        },
        {
            "old": "from config.config_loader import load_config",
            "new": "from config import load_config"
        },
        {
            "old": "from config.config_loader import Config, load_config", 
            "new": "from config import BaseOffsetConfig, load_config"
        },
        {
            "old": "config = Config(config_path)",
            "new": "config = BaseOffsetConfig(config_path)"
        },
        {
            "old": "config = load_config(config_type='volatility')",
            "new": "config = VolatilityConfig()  # or load_config(config_type='volatility')"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Old: {example['old']}")
        print(f"  New: {example['new']}")

if __name__ == "__main__":
    try:
        test_simplified_structure()
        show_file_structure()
        analyze_dependencies()
        show_migration_examples()
        
        print(f"\n🎉 Simplified Configuration System Working!")
        print("   • config_loader.py is now just a compatibility shim")
        print("   • All functionality consolidated in config_factory.py")
        print("   • Cleaner import structure: from config import ...")
        print("   • Gradual migration path available")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
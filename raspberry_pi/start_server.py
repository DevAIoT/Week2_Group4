#!/usr/bin/env python3
"""
Golf Swing Server Startup Script
Simple launcher for the golf swing ML prediction server
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Change to raspberry_pi directory
os.chdir(Path(__file__).parent)

try:
    from golf_swing_server import main
    import asyncio
    
    print("🏌️ Starting Golf Swing ML Server...")
    print("Make sure your ESP32 is connected via USB")
    print("Frontend should connect to ws://10.78.111.133:8765")
    print("\nPress Ctrl+C to stop the server\n")
    
    asyncio.run(main())
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n👋 Server stopped by user")
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)
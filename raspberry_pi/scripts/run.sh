#!/bin/bash

# Golf Swing Analyzer - Run Script
# This script starts the Raspberry Pi data processing system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Golf Swing Analyzer - Starting System${NC}"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    echo -e "${RED}Error: Must be run from raspberry_pi directory${NC}"
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed. Please install uv first.${NC}"
    echo "Installation: pip install uv"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Check Python dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found${NC}"
    exit 1
fi

# Install dependencies if needed
echo -e "${YELLOW}Installing/updating dependencies...${NC}"
uv sync

# Check if config file exists
if [ ! -f "config/config.yaml" ]; then
    echo -e "${RED}Error: config/config.yaml not found${NC}"
    exit 1
fi

# Check for required permissions (Bluetooth)
echo -e "${YELLOW}Checking Bluetooth permissions...${NC}"
if ! groups $USER | grep -q '\bbluetooth\b'; then
    echo -e "${YELLOW}Warning: User not in bluetooth group. You may need to run with sudo or add user to bluetooth group${NC}"
    echo "To add user to bluetooth group: sudo usermod -a -G bluetooth \$USER"
fi

# Start the system
echo -e "${GREEN}Starting Golf Swing Analyzer...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Run with uv
exec uv run python src/main.py
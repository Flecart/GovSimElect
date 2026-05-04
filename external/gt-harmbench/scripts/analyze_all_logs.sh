#!/bin/bash
# Script to run default analysis (accuracy, welfare, heatmap) for all logs in a directory

set -e  # Exit on error

# Default values
LOG_DIR="${1:-logs}"
OUTPUT_DIR="${2:-assets}"
PREFIX="${3:-}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Analyzing all logs in: ${LOG_DIR}${NC}"
echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}"
if [ -n "$PREFIX" ]; then
    echo -e "${GREEN}Prefix: ${PREFIX}${NC}"
fi
echo ""

# Check if directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo -e "${RED}Error: Directory ${LOG_DIR} does not exist${NC}"
    exit 1
fi

# Count files
EVAL_COUNT=$(find "$LOG_DIR" -maxdepth 1 -name "*.eval" -type f | wc -l)
DIR_COUNT=$(find "$LOG_DIR" -maxdepth 1 -type d ! -path "$LOG_DIR" | wc -l)
TOTAL=$((EVAL_COUNT + DIR_COUNT))

if [ "$TOTAL" -eq 0 ]; then
    echo -e "${YELLOW}No log files or directories found in ${LOG_DIR}${NC}"
    exit 0
fi

echo -e "${GREEN}Found ${EVAL_COUNT} .eval files and ${DIR_COUNT} directories${NC}"
echo ""

# Process .eval files
if [ "$EVAL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}Processing .eval files...${NC}"
    for eval_file in "$LOG_DIR"/*.eval; do
        if [ -f "$eval_file" ]; then
            filename=$(basename "$eval_file")
            echo -e "${YELLOW}Processing: ${filename}${NC}"
            
            if [ -n "$PREFIX" ]; then
                PYTHONPATH=. uv run python -m eval.analysis.cli \
                    --log-path "$eval_file" \
                    --log-type eval \
                    --output-dir "$OUTPUT_DIR" \
                    --prefix "$PREFIX" \
                    --plot accuracy \
                    --plot welfare \
                    --plot heatmap
            else
                PYTHONPATH=. uv run python -m eval.analysis.cli \
                    --log-path "$eval_file" \
                    --log-type eval \
                    --output-dir "$OUTPUT_DIR" \
                    --plot accuracy \
                    --plot welfare \
                    --plot heatmap
            fi
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Completed: ${filename}${NC}"
            else
                echo -e "${RED}✗ Failed: ${filename}${NC}"
            fi
            echo ""
        fi
    done
fi

# Process directories
if [ "$DIR_COUNT" -gt 0 ]; then
    echo -e "${GREEN}Processing directories...${NC}"
    for log_dir in "$LOG_DIR"/*/; do
        if [ -d "$log_dir" ] && [ "$(basename "$log_dir")" != "." ] && [ "$(basename "$log_dir")" != ".." ]; then
            dirname=$(basename "$log_dir")
            # Check if it looks like a log directory (has samples subdirectory)
            if [ -d "$log_dir/samples" ]; then
                echo -e "${YELLOW}Processing directory: ${dirname}${NC}"
                
                if [ -n "$PREFIX" ]; then
                    PYTHONPATH=. uv run python -m eval.analysis.cli \
                        --log-path "$log_dir" \
                        --log-type dir \
                        --output-dir "$OUTPUT_DIR" \
                        --prefix "$PREFIX" \
                        --plot accuracy \
                        --plot welfare \
                        --plot heatmap
                else
                    PYTHONPATH=. uv run python -m eval.analysis.cli \
                        --log-path "$log_dir" \
                        --log-type dir \
                        --output-dir "$OUTPUT_DIR" \
                        --plot accuracy \
                        --plot welfare \
                        --plot heatmap
                fi
                
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}✓ Completed: ${dirname}${NC}"
                else
                    echo -e "${RED}✗ Failed: ${dirname}${NC}"
                fi
                echo ""
            else
                echo -e "${YELLOW}Skipping ${dirname} (no samples/ subdirectory)${NC}"
            fi
        fi
    done
fi

echo -e "${GREEN}All done!${NC}"


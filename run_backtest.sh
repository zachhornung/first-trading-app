#!/bin/bash

# IBKR Trading Platform - Backtest Runner
# Usage: ./run_backtest.sh [options]

set -e

# Default values
SYMBOLS="AAPL,MSFT,GOOGL"
START_DATE="2023-01-01"
END_DATE="2023-12-31"
CAPITAL="100000"
BAR_SIZE="1 day"
OUTPUT_DIR="backtest_results"
VERBOSE=false
LOG_LEVEL="info"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
IBKR Historical Backtest Runner

USAGE:
    ./run_backtest.sh [OPTIONS]

OPTIONS:
    -s, --symbols SYMBOLS       Comma-separated list of symbols (default: AAPL,MSFT,GOOGL)
    -f, --from DATE             Start date YYYY-MM-DD (default: 2023-01-01)
    -t, --to DATE               End date YYYY-MM-DD (default: 2023-12-31)
    -c, --capital AMOUNT        Initial capital (default: 100000)
    -b, --bar-size SIZE         Bar size: "1 day", "1 hour", "30 mins", etc. (default: "1 day")
    -o, --output DIR            Output directory (default: backtest_results)
    -v, --verbose               Enable verbose logging
    -l, --log-level LEVEL       Log level: error, warn, info, debug (default: info)
    -h, --help                  Show this help message

EXAMPLES:
    # Basic backtest
    ./run_backtest.sh --symbols AAPL --from 2023-06-01 --to 2023-12-31

    # Multiple symbols with verbose output
    ./run_backtest.sh --symbols AAPL,MSFT,TSLA --verbose

    # High-frequency backtest
    ./run_backtest.sh --symbols AAPL --bar-size "15 mins" --from 2023-11-01 --to 2023-11-30

    # Large portfolio backtest
    ./run_backtest.sh --symbols AAPL,MSFT,GOOGL,TSLA,NVDA,AMZN,META --capital 500000

PREREQUISITES:
    1. IBKR TWS or Gateway must be running
    2. API access must be enabled in IBKR settings
    3. Proper configuration in config.toml

For setup instructions, see: IBKR_SETUP.md
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        -f|--from)
            START_DATE="$2"
            shift 2
            ;;
        -t|--to)
            END_DATE="$2"
            shift 2
            ;;
        -c|--capital)
            CAPITAL="$2"
            shift 2
            ;;
        -b|--bar-size)
            BAR_SIZE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            LOG_LEVEL="debug"
            shift
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate dates
if ! date -d "$START_DATE" >/dev/null 2>&1; then
    print_error "Invalid start date: $START_DATE (use YYYY-MM-DD format)"
    exit 1
fi

if ! date -d "$END_DATE" >/dev/null 2>&1; then
    print_error "Invalid end date: $END_DATE (use YYYY-MM-DD format)"
    exit 1
fi

if [[ $(date -d "$START_DATE" +%s) -ge $(date -d "$END_DATE" +%s) ]]; then
    print_error "Start date must be before end date"
    exit 1
fi

# Validate capital
if ! [[ "$CAPITAL" =~ ^[0-9]+$ ]] || [[ "$CAPITAL" -lt 1000 ]]; then
    print_error "Capital must be a number >= 1000"
    exit 1
fi

# Calculate date range
start_timestamp=$(date -d "$START_DATE" +%s)
end_timestamp=$(date -d "$END_DATE" +%s)
days_diff=$(( (end_timestamp - start_timestamp) / 86400 ))

# Show banner
cat << 'EOF'

╔══════════════════════════════════════════════════════════════╗
║                 IBKR BACKTEST RUNNER                        ║
║                                                              ║
║  Automated historical backtesting with Interactive Brokers  ║
╚══════════════════════════════════════════════════════════════╝

EOF

# Show configuration
print_info "Backtest Configuration:"
echo "  📊 Symbols: $SYMBOLS"
echo "  📅 Period: $START_DATE to $END_DATE ($days_diff days)"
echo "  📈 Bar Size: $BAR_SIZE"
echo "  💰 Initial Capital: \$$(printf "%'d" $CAPITAL)"
echo "  📁 Output: $OUTPUT_DIR"
echo "  🔍 Log Level: $LOG_LEVEL"

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    print_error "Cargo not found. Please install Rust and Cargo."
    exit 1
fi

# Check if project builds
print_info "Checking project build status..."
if ! cargo check --quiet; then
    print_error "Project failed to build. Please fix compilation errors."
    exit 1
fi
print_success "Project builds successfully"

# Check if IBKR is likely running (basic port check)
if command -v nc &> /dev/null; then
    if nc -z 127.0.0.1 7497 2>/dev/null; then
        print_success "IBKR appears to be running on port 7497 (paper trading)"
    elif nc -z 127.0.0.1 7496 2>/dev/null; then
        print_success "IBKR appears to be running on port 7496 (live trading)"
    else
        print_warning "IBKR doesn't appear to be running. Make sure TWS or Gateway is started."
        print_info "Continuing anyway - the application will provide more detailed error messages..."
    fi
else
    print_warning "Cannot check IBKR connection (nc not available)"
fi

# Create output directory
if [[ ! -d "$OUTPUT_DIR" ]]; then
    print_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Build command arguments
CMD_ARGS=(
    "--symbols" "$SYMBOLS"
    "--start-date" "$START_DATE"
    "--end-date" "$END_DATE"
    "--capital" "$CAPITAL"
    "--bar-size" "$BAR_SIZE"
    "--output" "$OUTPUT_DIR"
)

if [[ "$VERBOSE" == "true" ]]; then
    CMD_ARGS+=("--verbose")
fi

# Set environment variables
export RUST_LOG="$LOG_LEVEL"
export RUST_BACKTRACE=1

# Show final command
print_info "Running backtest command:"
echo "RUST_LOG=$LOG_LEVEL cargo run --bin backtest -- ${CMD_ARGS[*]}"
echo

# Run the backtest
print_info "Starting historical backtest..."
echo "yes" | cargo run --bin backtest -- "${CMD_ARGS[@]}"

# Check if backtest was successful
if [[ $? -eq 0 ]]; then
    print_success "Backtest completed successfully!"
    
    # Show output files if they exist
    if [[ -d "$OUTPUT_DIR" ]]; then
        print_info "Generated files:"
        find "$OUTPUT_DIR" -type f -name "*.csv" -o -name "*.json" -o -name "*.html" | head -10 | while read -r file; do
            echo "  📄 $file"
        done
        
        # Show file count if there are many
        file_count=$(find "$OUTPUT_DIR" -type f | wc -l)
        if [[ $file_count -gt 10 ]]; then
            echo "  ... and $((file_count - 10)) more files"
        fi
    fi
    
    print_info "Check the output directory for detailed results: $OUTPUT_DIR"
else
    print_error "Backtest failed. Check the error messages above."
    print_info "Common issues:"
    echo "  • IBKR TWS/Gateway not running"
    echo "  • API access not enabled"
    echo "  • Invalid symbols or date ranges"
    echo "  • Network connectivity issues"
    echo
    print_info "See IBKR_SETUP.md for detailed setup instructions"
    exit 1
fi

# Offer to open results
if command -v open &> /dev/null && [[ -d "$OUTPUT_DIR" ]]; then
    echo
    read -p "Open results directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$OUTPUT_DIR"
    fi
fi

print_success "Backtest runner completed!"
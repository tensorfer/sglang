#!/bin/bash

# Default parameter values
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="30000"
DEFAULT_MODEL_PATH=""
DEFAULT_DATASET_PATH="ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_ENABLE_EXPLICIT_KVCACHE="--enable-explicit-kvcache"
DEFAULT_ENABLE_HICACHE=""

DEFAULT_REQUEST_LENGTH="8192"
DEFAULT_SUB_QUESTION_LENGTH="512"
DEFAULT_OUTPUT_LENGTH="512"
DEFAULT_NUM_CLIENTS="16"
DEFAULT_NUM_ROUNDS="8"
DEFAULT_MAX_PARALLEL="16"

# Function: Display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --host HOST                     Server host (default: $DEFAULT_HOST)"
    echo "  -p, --port PORT                     Server port (default: $DEFAULT_PORT)"
    echo "  -m, --model-path PATH               Model path (default: $DEFAULT_MODEL_PATH)"
    echo "  -d, --dataset-path PATH             Dataset path (default: $DEFAULT_DATASET_PATH)"
    echo "  -b, --background                    Run in background"
    echo "  --disable-explicit-kvcache          Disable explicit kvcache"
    echo "  --explicit-kvcache-backend          Backend for explicit kvcache (default: file)"
    echo "  --explicit-kvcache-backend-config   Config for explicit kvcache backend"
    echo "  --enable-hicache                    Enable hierarchical cache"
    echo "  --help                              Show this help message"
    echo ""
    echo "Env Variables:"
    echo "  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR: Cache directory for file backend"
    echo ""
    echo "Examples:"
    echo "  $0 -h 0.0.0.0 -p 30001 -b"
    echo "  $0 --model-path /path/to/model --background"
    echo "  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/dev/shm/hicache $0 -h 0.0.0.0 -p 30001 -b"
}

# Parse command line arguments
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
MODEL_PATH="$DEFAULT_MODEL_PATH"
DATASET_PATH="$DEFAULT_DATASET_PATH"
BACKGROUND=false
REQUEST_LENGTH="$DEFAULT_REQUEST_LENGTH"
SUB_QUESTION_LENGTH="$DEFAULT_SUB_QUESTION_LENGTH"
OUTPUT_LENGTH="$DEFAULT_OUTPUT_LENGTH"
NUM_CLIENTS="$DEFAULT_NUM_CLIENTS"
NUM_ROUNDS="$DEFAULT_NUM_ROUNDS"
MAX_PARALLEL="$DEFAULT_MAX_PARALLEL"
ENABLE_EXPLICIT_KVCACHE="$DEFAULT_ENABLE_EXPLICIT_KVCACHE"
EXPLICIT_KVCACHE_BACKEND="file"
EXPLICIT_KVCACHE_BACKEND_CONFIG=""
ENABLE_HICACHE="$DEFAULT_ENABLE_HICACHE"


# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        --disable-explicit-kvcache)
            ENABLE_EXPLICIT_KVCACHE=""
            shift
            ;;
        --explicit-kvcache-backend)
            EXPLICIT_KVCACHE_BACKEND="$2"
            shift 2
            ;;
        --explicit-kvcache-backend-config)
            EXPLICIT_KVCACHE_BACKEND_CONFIG="$2"
            shift 2
            ;;
        --enable-hicache)
            ENABLE_HICACHE="--enable-hierarchical-cache"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Display configuration information
echo "=== Configuration Parameters ==="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Request Length: $REQUEST_LENGTH"
echo "Sub-question Length: $SUB_QUESTION_LENGTH"
echo "Output Length: $OUTPUT_LENGTH"
echo "Number of Clients: $NUM_CLIENTS"
echo "Number of Rounds: $NUM_ROUNDS"
echo "Maximum Parallel: $MAX_PARALLEL"
echo "Background Mode: $BACKGROUND"
echo "Explicit KVCache: $([[ -n "$ENABLE_EXPLICIT_KVCACHE" ]] && echo "enable" || echo "disable")"
echo "Explicit KVCache Backend: $EXPLICIT_KVCACHE_BACKEND"
echo "Explicit KVCache Backend Config: $EXPLICIT_KVCACHE_BACKEND_CONFIG"
echo "Hierarchical Cache: $([[ -n "$ENABLE_HICACHE" ]] && echo "enable" || echo "disable")"
echo "================================"

# Server launch command
SERVER_CMD="python3 -m sglang.launch_server \
    --host $HOST \
    --port $PORT \
    --model-path $MODEL_PATH \
    $ENABLE_HICACHE \
    --mem-fraction-static 0.8 \
    $ENABLE_EXPLICIT_KVCACHE"

# Benchmark command
BENCH_CMD="python3 bench_multiturn.py \
    --host $HOST \
    --port $PORT \
    --model-path $MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --request-length $REQUEST_LENGTH \
    --sub-question-input-length $SUB_QUESTION_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --num-clients $NUM_CLIENTS \
    --num-rounds $NUM_ROUNDS \
    --max-parallel $MAX_PARALLEL \
    $ENABLE_EXPLICIT_KVCACHE \
    --explicit-kvcache-backend $EXPLICIT_KVCACHE_BACKEND \
    --explicit-kvcache-backend-config '$EXPLICIT_KVCACHE_BACKEND_CONFIG'"

# Function to run commands in background
run_in_background() {
    local cmd="$1"
    local log_file="$2"

    echo "Running in background: $cmd"
    echo "Output logged to: $log_file"

    # Use nohup to run in background and redirect output to log file
    nohup bash -c "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    echo "Process ID: $pid"
    echo $pid > "${log_file%.log}.pid"
    return $pid
}

# Main execution logic
main() {
    echo "Starting benchmark test..."

    # Check if necessary files exist
    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Error: Dataset file does not exist: $DATASET_PATH"
        exit 1
    fi

    if [[ ! -d "$(dirname "$MODEL_PATH")" ]]; then
        echo "Warning: Model path may not exist: $MODEL_PATH"
    fi

    # Create log directory
    LOG_DIR="logs"
    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SERVER_LOG="$LOG_DIR/${TIMESTAMP}_server.log"
    BENCH_LOG="$LOG_DIR/${TIMESTAMP}_bench.log"

    # Start server
    echo "Starting server..."
    if [[ "$BACKGROUND" == true ]]; then
        SERVER_PID=$(run_in_background "$SERVER_CMD" "$SERVER_LOG")
        echo "Server started, PID: $SERVER_PID"

        # Wait for server to start
        echo "Waiting for server to start..."
        sleep 60
    else
        echo "Starting server in foreground..."
        eval $SERVER_CMD &
        SERVER_PID=$!
        echo "Server PID: $SERVER_PID"
        sleep 60
    fi

    # Run benchmark
    echo "Running benchmark..."
    if [[ "$BACKGROUND" == true ]]; then
        BENCH_PID=$(run_in_background "$BENCH_CMD" "$BENCH_LOG")
        echo "Benchmark started, PID: $BENCH_PID"
        echo "Benchmark is running in background, check log file: $BENCH_LOG"
    else
        echo "Running benchmark in foreground..."
        eval $BENCH_CMD
    fi

    # If not running in background, wait for server process to finish
    if [[ "$BACKGROUND" == false ]]; then
        wait $SERVER_PID
    fi
}

# Execute main function
main "$@"

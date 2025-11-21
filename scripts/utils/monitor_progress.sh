#!/bin/bash
# Monitor BAYC batch analysis progress

LOG_FILE="/Users/qingshen/Desktop/opensea/opensea_pipeline/bayc_analysis.log"
JSON_FILE="/Users/qingshen/Desktop/opensea/opensea_pipeline/bayc_metadata.json"

echo "======================================"
echo "BAYC Batch Analysis - Progress Monitor"
echo "======================================"
echo ""

# Check if process is running
if pgrep -f "python.*batch_analyze_bayc.py" > /dev/null; then
    echo "✅ Status: RUNNING"
    PID=$(pgrep -f "python.*batch_analyze_bayc.py")
    echo "   Process ID: $PID"
else
    echo "⏸️  Status: NOT RUNNING or COMPLETED"
fi

echo ""
echo "📊 Progress:"
if [ -f "$LOG_FILE" ]; then
    # Count processed images
    PROCESSED=$(grep -c "Processing BAYC #" "$LOG_FILE")
    echo "   Images processed: $PROCESSED / 200"
    
    # Count successful
    SUCCESS=$(grep -c "✓ Background:" "$LOG_FILE")
    echo "   Successful: $SUCCESS"
    
    # Count errors
    ERRORS=$(grep -c "✗ Error:" "$LOG_FILE")
    echo "   Errors: $ERRORS"
    
    # Show last processed
    LAST=$(grep "Processing BAYC #" "$LOG_FILE" | tail -1)
    echo ""
    echo "🔄 Last processed:"
    echo "   $LAST"
    
    # Show last few lines
    echo ""
    echo "📝 Recent activity (last 10 lines):"
    tail -10 "$LOG_FILE" | sed 's/^/   /'
else
    echo "   ⚠️  Log file not found"
fi

echo ""
echo "📁 Output file: $JSON_FILE"
if [ -f "$JSON_FILE" ]; then
    # Count entries in JSON
    ENTRIES=$(python3 -c "import json; print(len(json.load(open('$JSON_FILE'))))" 2>/dev/null || echo "N/A")
    echo "   JSON entries: $ENTRIES"
else
    echo "   ⚠️  Output file not yet created"
fi

echo ""
echo "======================================"
echo "Commands:"
echo "  Watch live: tail -f $LOG_FILE"
echo "  Check progress: bash monitor_progress.sh"
echo "  Kill process: kill $PID"
echo "======================================"

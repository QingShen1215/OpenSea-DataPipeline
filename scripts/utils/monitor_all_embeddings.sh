#!/bin/bash
# Monitor batch embedding generation progress

echo "======================================"
echo "BAYC Full Collection - Progress Monitor"
echo "======================================"
echo ""

# Count converted images
CONVERTED_DIR="/Users/qingshen/Desktop/opensea/bayc_images/converted_temp"
CONVERTED_COUNT=$(ls "$CONVERTED_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
TOTAL_IMAGES=9998

echo "📊 Image Conversion Progress:"
echo "   Converted: $CONVERTED_COUNT / $TOTAL_IMAGES"

# Calculate percentage
PERCENTAGE=$(echo "scale=1; $CONVERTED_COUNT * 100 / $TOTAL_IMAGES" | bc)
echo "   Progress: ${PERCENTAGE}%"

# Estimate time remaining
if [ $CONVERTED_COUNT -gt 0 ]; then
    REMAINING=$((TOTAL_IMAGES - CONVERTED_COUNT))
    TIME_REMAINING=$((REMAINING / 17))  # ~17 images/second
    echo "   Estimated time remaining: ~${TIME_REMAINING} seconds ($(echo "scale=1; $TIME_REMAINING / 60" | bc) minutes)"
fi

echo ""
echo "📁 Output directory: $CONVERTED_DIR"
echo ""

# Check if embeddings exist
EMBED_FILE="/Users/qingshen/Desktop/opensea/opensea_pipeline/embeddings/bayc_embeddings_clip.npz"
if [ -f "$EMBED_FILE" ]; then
    EMBED_SIZE=$(ls -lh "$EMBED_FILE" | awk '{print $5}')
    echo "✓ Embeddings file exists: $EMBED_SIZE"
else
    echo "⏳ Embeddings not yet generated (waiting for conversion to complete)"
fi

echo ""
echo "======================================"
echo "Commands:"
echo "  Watch progress: watch -n 5 bash monitor_all_embeddings.sh"
echo "  Check terminal: press Ctrl+C to stop monitoring"
echo "======================================"

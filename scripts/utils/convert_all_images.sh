#!/bin/bash
# Fast parallel conversion of ALL BAYC images

ORIGINAL_DIR="/Users/qingshen/Desktop/opensea/bayc_images"
CONVERTED_DIR="/Users/qingshen/Desktop/opensea/bayc_images/converted_temp"

echo "======================================"
echo "Converting ALL BAYC Images"
echo "======================================"
echo ""

# Count images
TOTAL=$(ls "$ORIGINAL_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo "📊 Total images to convert: $TOTAL"

# Get CPU cores
NPROC=$(sysctl -n hw.ncpu)
WORKERS=$((NPROC / 2))
echo "🚀 Using $WORKERS parallel workers"
echo "⏱️  Estimated time: ~$(echo "scale=1; $TOTAL / $WORKERS / 10" | bc) minutes"
echo ""

# Convert all images in parallel using xargs
cd "$ORIGINAL_DIR"
ls *.png | xargs -P $WORKERS -I {} sh -c '
    base=$(basename "{}" .png)
    magick "'"$ORIGINAL_DIR"'/{}" "'"$CONVERTED_DIR"'/${base}_converted.png" 2>/dev/null && echo -n "."
'

echo ""
echo ""

# Count results
CONVERTED=$(ls "$CONVERTED_DIR"/*_converted.png 2>/dev/null | wc -l | tr -d ' ')
echo "✓ Conversion complete!"
echo "  Converted: $CONVERTED / $TOTAL"
echo ""

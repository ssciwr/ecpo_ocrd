#!/bin/bash

#
# An example workflow developed while looking at Jingbao images.
#

# Step 0.1: Image Enhancement
ocrd-skimage-denoise-raw -I OCR-D-IMG -O OCR-D-DENOISED
ocrd-skimage-normalize -I OCR-D-DENOISED -O OCR-D-NORMALIZED

# Step 0.2: Font detection does not apply (or there are no models available)

# Step 1: Binarization (Page Level)
ocrd-sbb-binarize -I OCR-D-NORMALIZED -O OCR-D-BIN -P model default-2021-03-09

# Step 2: Cropping (Page Level) not required for Jingbao. Applying it detected non-existent
#         rulers and arbitrarily cropped the images.

# Step 3: Binarization (Page Level) does not apply if Step 2 was skipped.

# Step 4: Denoising (Page Level)
ocrd-cis-ocropy-denoise -I OCR-D-BIN -O OCR-D-DENOISE

# Step 5: Deskewing (Page Level)
ocrd-cis-ocropy-deskew -I OCR-D-DENOISE -O OCR-D-DESKEW-PAGE -P level-of-operation page

# Step 6: Dewarping (Page Level)
# TODO: This tool is missing in my Docker container. Might be related to the "GPU required" disclaimer
#       in the documentation. Are there GPU-enabled Docker images for OCR-D tools?
# ocrd-anybaseocr-dewarp -I OCR-D-DESKEW-PAGE -O OCR-D-DEWARP-PAGE

# Step 7: Region segmentation
# Note: Eynollah does not work for me here, as it does not download one of the required models.
# ocrd-eynollah-segment -I OCR-D-DESKEW-PAGE -O OCR-D-SEG -P models default
ocrd-paddleocr-segment -I OCR-D-IMG -O OCR-D-SEG -P threshold 30 -P layout_merge_bboxes_mode large

# Output visualization
ocrd-regions-to-labelstudio -I OCR-D-SEG -O OCR-D-LS

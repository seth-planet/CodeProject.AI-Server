#!/bin/bash

# Development mode setup script ::::::::::::::::::::::::::::::::::::::::::::::
#
#                            Object Detection (YOLOv8)
#
# This script is called from the ObjectDetectionYOLOv8 directory using: 
#
#    bash ../../setup.sh
#
# The setup.sh script will find this install.sh file and execute it.
#
# For help with install scripts, notes on variables and methods available, tips,
# and explanations, see /src/modules/install_script_help.md

if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../setup.sh"
    echo
    exit 1 
fi

# Download the models from CodeProject's models folder and store in /assets
getFromServer "models/" "models-yolo8-pt.zip"  "assets" "Downloading Standard YOLO models..."

# TODO: Check assets created and has files
# module_install_errors=...

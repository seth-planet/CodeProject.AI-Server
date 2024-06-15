#!/bin/bash

# Development mode setup script ::::::::::::::::::::::::::::::::::::::::::::::
#
#                            .NET YOLO Object Detection
#
# This script is called from the ObjectDetectionYOLOv5Net directory using: 
#
#    bash ../../../setup.sh
#
# The setup.sh script will find this install.sh file and execute it.
#
# For help with install scripts, notes on variables and methods available, tips,
# and explanations, see /src/modules/install_script_help.md

if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../../setup.sh"
    echo
    exit 1 
fi

# Pull down the correct .NET executable for this module
if [ "${executionEnvironment}" = "Production" ] || [ "${launchedBy}" = "server" ]; then
    imageName="${moduleId}-${moduleVersion}.zip"
    getFromServer "binaries/" "${imageName}" "bin" "Downloading ${imageName}..."
else
    pushd "$moduleDirPath" >/dev/null
    writeLine "Building project..." "$color_info"
    dotnet build -c Debug -o "${moduleDirPath}/bin/Debug/net7.0" >/dev/null
    popd >/dev/null
fi

# Download the models and store in /assets
getFromServer "models/" "objectdetection-coco-yolov8-onnx-m.zip"        "assets"        "Downloading YOLO ONNX models..."

# Install ONNX runtime
if [ "$os" = "macos" ]; then
    writeLine "Installing ONNX runtime "
    if [ "${platform}" = "macos" ]; then
        brew list onnxruntime >/dev/null 2>/dev/null || brew install onnxruntime >/dev/null 2>/dev/null &
    elif [ "${platform}" = "macos-arm64" ]; then
        arch -x86_64 /usr/local/bin/brew list onnxruntime || arch -x86_64 /usr/local/bin/brew install onnxruntime
    fi
    writeLine "done" "$color_success"
fi

# TODO: Check assets created and has files
# module_install_errors=...

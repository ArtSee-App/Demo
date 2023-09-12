#!/bin/bash

# Set the path to your Jupyter Notebook executable
JUPYTER_EXECUTABLE="jupyter"

# Set the path to your Python notebook file
NOTEBOOK_FILE="artsee-demo.ipynb"

# Convert the notebook to a script
$JUPYTER_EXECUTABLE nbconvert --to script $NOTEBOOK_FILE

# Run the generated Python script
python "${NOTEBOOK_FILE%.*}.py"

# Remove the generated Python script
rm "${NOTEBOOK_FILE%.*}.py"

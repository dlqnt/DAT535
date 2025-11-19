#!/bin/bash

# --- Configuration ---
YEARS_TO_DOWNLOAD=$(seq 2020 2024)
TAXI_TYPES="yellow green"
BASE_URL="https://d37ci6vzurychx.cloudfront.net/trip-data"
TARGET_DIR="raw_tlc_data" 

# --- Script Logic ---
echo "Starting TLC data download..."
# Create the main directory if it doesn't exist.
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"
echo "Data will be saved in: $(pwd)"

for type in ${TAXI_TYPES}; do
  mkdir -p "${type}"
  cd "${type}"
  echo "--- Downloading ${type} taxi data for missing files ---"

  for year in ${YEARS_TO_DOWNLOAD}; do
    for month in {1..12}; do
      FMONTH=$(printf "%02d" ${month})
      FILENAME="${type}_tripdata_${year}-${FMONTH}.parquet"
      
      # Check if the file already exists locally before doing anything else.
      if [ -f "${FILENAME}" ]; then
         echo "${FILENAME} already exists, skipping."
        continue 
      fi
      # --- END OF THE LOCAL FILE CHECK ---

      URL="${BASE_URL}/${FILENAME}"
      echo "Attempting to download ${URL}"
      
      # only run wget if the file is confirmed to be missing.
      wget -c "${URL}" || echo "File not found or download failed for ${FILENAME}. Continuing."
      
      # Wait for a random number of seconds (1 to 3) to be polite.
      SLEEP_TIME=$((RANDOM % 3 + 1))
      echo "Sleeping for ${SLEEP_TIME} seconds..."
      sleep ${SLEEP_TIME}
      
    done
  done
  
  cd ..
done

echo "--- All downloads complete! ---"


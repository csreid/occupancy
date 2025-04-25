#!/bin/bash

# Run the Python script in the background
poetry run python mcap_to_sqlite.py &
PID=$!

# Monitor memory usage
while kill -0 $PID 2>/dev/null; do
  MEM_KB=$(ps -o rss= -p $PID)
  MEM_MB=$((MEM_KB / 1024))

  if [ $MEM_MB -gt 4000 ]; then
    echo "Memory limit exceeded, killing process"
    kill $PID
    break
  fi

  sleep 1
done

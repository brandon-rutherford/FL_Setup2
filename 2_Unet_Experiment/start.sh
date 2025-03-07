#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_clients>"
    exit 1
fi

# Get the number of clients from the argument
num_clients=$1

# Start the server in the background
python server.py &
server_pid=$!
echo "Server started with PID $server_pid"

# Start the clients in the background
for ((i=0; i<num_clients; i++)); do
    python client.py $i &
    echo "Started client $i"
done

# Wait for the server process to finish
wait $server_pid
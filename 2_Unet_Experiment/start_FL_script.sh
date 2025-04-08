#!/bin/bash

# When the script exits (or is interrupted), kill all child processes.
trap "echo 'Killing all child processes...'; kill $(jobs -p)" SIGINT SIGTERM EXIT

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

# Function to launch a client in a new terminal window.
launch_client() {
    client_id=$1
    # Check for gnome-terminal, then xterm
    if command -v gnome-terminal >/dev/null; then
        gnome-terminal -- bash -c "python client.py $client_id $num_clients; exec bash"
    elif command -v xterm >/dev/null; then
        xterm -hold -e "python client.py $client_id $num_clients" &
    else
        echo "No supported terminal emulator found. Launching client $client_id in background."
        python client.py $client_id $num_clients &
    fi
}

# Give the server time to start up (adjust sleep duration if needed)
sleep 5

# Launch each client in its own terminal
for (( i=0; i<num_clients; i++ )); do
    echo "Starting client with id $i in a new terminal..."
    launch_client $i
done

# Wait for the server process to finish.
wait $server_pid

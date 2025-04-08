NUM_CLIENTS = 10 # this is just for saving purposes.

# ** MAKE SURE THIS IS SAME AS BASH SCRIPT NUMBER **

# Max or explicitly defined number of epochs per round
NUM_EPOCHS = 1

# Early stopping patience for clients. 0= no early stopping, -1 = early stopping relative to the round number.
EPOCH_EARLY_STOPPING_PATIENCE = 0

# Batch size on each of the clients
BATCH_SIZE = 1

# How many rounds of federated learning can pass without aggregated DICE not improving before early stopping.
NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP = 1

# Client dropout
CLIENT_PARTICIPATION_FRACTION = 1 # Represents what fraction of clients will be trained and used for eval each round

# max rounds
NUM_ROUNDS= 25

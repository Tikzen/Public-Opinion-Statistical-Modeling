import random

MODEL_NAME = "SEIR"

STATES = ["S", "E", "I", "R"]


def step(G, state, infection_prob, recovery_prob, blocked_nodes=None):
    if blocked_nodes is None:
        blocked_nodes = set()

    new_state = state.copy()

    for node in G.nodes():

        if state[node] == "I":

            if node not in blocked_nodes:
                for neighbor in G.neighbors(node):

                    if state[neighbor] == "S":
                        if random.random() < infection_prob:
                            new_state[neighbor] = "E"

            if random.random() < recovery_prob:
                new_state[node] = "R"

        elif state[node] == "E":

            if random.random() < 0.5:
                new_state[node] = "I"

    return new_state
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from sympy import symbols, simplify, sin, cos


class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)


# Define operators
two_way_operators = ['*', '-', '+']
one_way_operators = ['sin', 'cos']
two_ways_operators = ['^', '/', '*', '-', '+']
one_ways_operators = ['tan', 'cot', 'log']


# Function to create a random tree
def create_random_tree(depth, max_depth):
    if depth == max_depth or random.random() < 0.3:
        if random.random() < 0.2:
            return Node(value='x')  # Variable
        else:
            return Node(value=str(random.randint(1, 9)))  # Random integer between 1 and 9
    else:

        if random.choice([True, False]) and depth < max_depth - 1:
            node = Node(value=random.choice(one_way_operators))
            node.add_child(create_random_tree(depth + 1, max_depth))
        else:

            node = Node(value=random.choice(two_way_operators))
        for _ in range(2):
            node.add_child(create_random_tree(depth + 1, max_depth))
    return node


# Function to print the tree
def print_tree(node, indent=0):
    if node:
        print('  ' * indent + str(node.value))
        for child in node.children:
            print_tree(child, indent + 1)


# Fitness Function
def fitness(tree, x):
    expression_result = evaluate_tree(tree, x)
    expression_result = round(expression_result, 4)
    # target_result = (9 * x) + 5 + (4 * x * x)  # + 5 * np.cos(x) + np.sin(x)
    target_result = 2 * x * np.sin(x) + (9 * x) + 5 + (4 * x * x)
    target_result = round(target_result, 4)
    # print("tree fitness: ",expression_result,"  ",target_result)
    return abs(expression_result - target_result)


def get_all_nodes(tree):
    nodes = [tree]
    for child in tree.children:
        nodes.extend(get_all_nodes(child))
    return nodes


# Evaluate the expression tree for a given x
def evaluate_tree(node, x):
    if node.value == 'x':
        return x
    elif node.value.isdigit():
        return int(node.value)
    elif node.value in ['sin', 'cos']:
        child_result = evaluate_tree(node.children[0], x)
        if isinstance(child_result, (int, float)):
            if node.value == 'sin':
                return np.sin(child_result)
            elif node.value == 'cos':
                return np.cos(child_result)
            else:
                return np.inf
        else:
            return np.inf

    else:
        left_result = evaluate_tree(node.children[0], x)
        right_result = evaluate_tree(node.children[1], x)

        if node.value == '^':
            if left_result < 0 and not isinstance(right_result, int):
                return 0
            elif left_result == 0 and right_result < 0:
                return np.inf
            else:
                try:
                    return left_result ** right_result
                except OverflowError:
                    return np.inf
        elif node.value == '/':
            # Check for division by zero
            if right_result == 0:
                return 0
            else:
                return left_result / right_result
        elif node.value == '*':
            return left_result * right_result
        elif node.value == '-':
            return left_result - right_result
        elif node.value == '+':
            return left_result + right_result


def crossover2(parent1, parent2):
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    node_parent1 = random.choice(get_all_nodes(parent1))
    node_parent2 = random.choice(get_all_nodes(parent2))

    node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
    node_parent1.children, node_parent2.children = node_parent2.children, node_parent1.children

    if node_parent1.value == 'x' and node_parent2.value == 'x':
        # If both selected nodes are variables, swap their values
        node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
        node_parent1.children, node_parent2.children = node_parent2.children, node_parent1.children
    elif node_parent1.value.isdigit() and node_parent2.value.isdigit():
        # If both selected nodes are numbers, swap their values
        node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
        node_parent1.children, node_parent2.children = node_parent2.children, node_parent1.children
        # If the selected node is a number, change it to another random number
    elif node_parent1.value in one_way_operators and node_parent2.value in one_way_operators:
        # If both selected nodes are one-way operators (e.g., sin, cos), swap their values
        node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
        node_parent1.children, node_parent2.children = node_parent2.children, node_parent1.children
        # If the selected node is a one-way operator (e.g., sin, cos), change it to the other operator
    elif node_parent1.value in two_way_operators and node_parent2.value in two_way_operators:
        # If both selected nodes are two-way operators, swap their values
        node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
        node_parent1.children, node_parent2.children = node_parent2.children, node_parent1.children
        # If the selected node is a two-way operator, change it to a different operator
    elif node_parent1.value == 'x' and node_parent2.value in one_way_operators:
        # If the selected node is a variable and the other node is a one-way operator, change the variable to a random number
        node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
        node_parent1.children, node_parent2.children = node_parent2.children, node_parent1.children
    else:
        node_parent1.value, node_parent2.value = node_parent1.value, node_parent2.value
        node_parent1.children, node_parent2.children = node_parent1.children, node_parent2.children

    return child1, child2


def crossover(parent1, parent2):
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    node_parent1 = random.choice(get_all_nodes(parent1))
    node_parent2 = random.choice(get_all_nodes(parent2))

    if node_parent1.value == 'x' and node_parent2.value == 'x':
        # If both selected nodes are variables, swap their values
        node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
    else:
        # Swap values and children for non-'x' nodes
        node_parent1.value, node_parent2.value = node_parent2.value, node_parent1.value
        node_parent1.children, node_parent2.children = node_parent2.children, node_parent1.children

    return child1, child2


def crossover3(parent1, parent2):
    # Make deep copies of the parents to avoid modifying the original trees
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # Randomly select a node from each parent
    node_parent1 = random.choice(get_all_nodes(parent1))
    node_parent2 = random.choice(get_all_nodes(parent2))

    # Swap subtrees between the selected nodes
    subtree_parent1 = copy.deepcopy(node_parent1)
    subtree_parent2 = copy.deepcopy(node_parent2)

    # Replace the corresponding subtrees in the children
    replace_subtree(child1, node_parent1, subtree_parent2)
    replace_subtree(child2, node_parent2, subtree_parent1)

    return child1, child2


def replace_subtree(tree, target_node, new_subtree):
    # Helper function to replace a subtree in the tree
    if tree == target_node:
        # If the tree is the target node, replace it with the new subtree
        tree.value = new_subtree.value
        tree.children = new_subtree.children
    else:
        # Recursively search for the target node in the children
        for i, child in enumerate(tree.children):
            replace_subtree(child, target_node, new_subtree)


# Mutation operation
def mutation(parent):
    available_operator = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    child = copy.deepcopy(parent)

    # Randomly select a node in the tree
    node_to_mutate = random.choice(get_all_nodes(child))

    # Mutate the selected node
    if node_to_mutate.value == 'x':
        if random.choice([True, False]):
            node_to_mutate.value = random.choice(available_operator)
        else:
            node_to_mutate.value = 'x'

    elif node_to_mutate.value.isdigit():
        if random.choice([True, False]):
            node_to_mutate.value = random.choice(available_operator)
        else:
            node_to_mutate.value = 'x'

    elif node_to_mutate.value in one_way_operators:
        node_to_mutate.value = 'sin' if node_to_mutate.value == 'cos' else 'cos'

    elif node_to_mutate.value in two_way_operators:
        # available_operators = ['^', '/', '*', '-', '+']
        available_operators = ['*', '-', '+']
        node_to_mutate.value = random.choice(available_operators)

    return child


def plot_fitness(generation, fitness_values):
    generations = range(1, generation + 2)
    plt.plot(generations, fitness_values, marker='o')
    plt.title('Fitness Evolution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.show()


def simplify_formula(node):
    if node.value == 'x' or node.value.isdigit():
        return node.value
    elif node.value in ['sin', 'cos']:
        return f"{node.value}({simplify_formula(node.children[0])})"
    else:
        left = simplify_formula(node.children[0])
        right = simplify_formula(node.children[1])
        return f"({left} {node.value} {right})"

def simplify_formula2(node):
    x = symbols('x')
    if node.value == 'x' or node.value.isdigit():
        return node.value
    elif node.value == 'sin':
        return sin(simplify_formula(node.children[0]))
    elif node.value == 'cos':
        return cos(simplify_formula(node.children[0]))
    else:
        left = simplify_formula(node.children[0])
        right = simplify_formula(node.children[1])
        expr = f"{left} {node.value} {right}"
        return simplify(expr)
# Generate Initial Population
number_population_trees = 1000
max_depth = 2
num_generations = 10
crossover_probability = 0.7
mutation_probability = 0.1
satisfactory_fitness = 1
x_values = np.linspace(-5, 5, 100)

initial_population_trees = [create_random_tree(0, max_depth) for _ in range(number_population_trees)]
best_individual = None
best_fitness_history = []
for generation in range(num_generations):
    fitness_values = []
    best_individuals = []
    z = 0
    # Iterate over each tree in the initial population
    for tree in initial_population_trees:
        total_fitness = 0.0
        # z = z + 1
        #  print("\n" + "-" * 20 + "\n")
        #  print(z)
        for x in np.linspace(-1000, 1000, 100000):
            try:
                current_fitness = fitness(tree, x)
            except OverflowError:
                current_fitness = np.inf

            total_fitness += current_fitness
        average_fitness = total_fitness / 100.0
        average_fitness = round(average_fitness, 4)
        fitness_values.append(average_fitness)

    # Check for infinite or NaN values in fitness_values
    if any(not np.isfinite(fit) for fit in fitness_values):
        # Handle the case where fitness values are problematic
        print(f"Error in generation {generation + 1}: Fitness values contain infinite or NaN values.")
        print("Problematic fitness values:", fitness_values)

    # Filter out problematic values and compute weights
    finite_fitness_values = [fit if np.isfinite(fit) else 0 for fit in fitness_values]
    total_fitness = sum(finite_fitness_values)
    weights = [fit / total_fitness for fit in finite_fitness_values]

    # Select parents based on fitness
    selected_parents = random.choices(initial_population_trees, weights=weights, k=len(initial_population_trees))

    # Print the best fitness in each generation
    best_fitness = min(fitness_values)
    best_fitness_history.append(best_fitness)

    # Update best individual
    best_individual_index = fitness_values.index(best_fitness)
    best_individual = copy.deepcopy(initial_population_trees[best_individual_index])
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
    print("\nFinal Tree:")
    print_tree(best_individual)
    print("\nSimplified Tree:")
    print(simplify_formula(best_individual))
    print("\nsuper Simplified Tree:")
    print(simplify_formula2(best_individual))
    if best_fitness < satisfactory_fitness or generation == num_generations - 1:
        print("Termination criteria met. Algorithm stopped.")
        break

    # Create the next generation using crossover and mutation
    next_generation = []
    for i in range(0, len(selected_parents), 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[i]

        # Crossover
        if random.random() < crossover_probability:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Mutation
        if random.random() < mutation_probability:
            child1 = mutation(child1)
        if random.random() < mutation_probability:
            child2 = mutation(child2)

        next_generation.extend([child1, child2])

    # Replace the old population with the new generation
    initial_population_trees = next_generation

print("\nFinal Tree:")
print_tree(best_individual)
print("\nSimplified Tree:")
print(simplify_formula(best_individual))
'''
plot_fitness(num_generations, best_fitness_history)
# Compare the predicted tree and the original function
x_values = np.linspace(-5, 5, 100)
original_function_values = [10 * x for x in x_values]
predicted_function_values = [evaluate_tree(best_individual, x) for x in x_values]

# Plot the original and predicted functions
plt.plot(x_values, original_function_values, label='Original Function')
plt.plot(x_values, predicted_function_values, label='Predicted Function')
plt.title('Comparison of Original and Predicted Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
'''

# Genetic Algorithm for Symbolic Regression

This project implements a genetic algorithm for symbolic regression, aiming to evolve mathematical expressions that approximate a target function. Here, we provide detailed explanations for the key functions involved in the algorithm.

## `create_random_tree(depth, max_depth)`

This function generates a random expression tree based on specified depth parameters. It recursively constructs the tree by randomly selecting operators and operands. The tree represents a mathematical expression.

```python
def create_random_tree(depth, max_depth):
```

## `fitness(tree, x)`

The fitness function evaluates the fitness of an expression tree by comparing its output to the target function's values. It calculates the absolute difference between the tree's result and the expected result for a given input x.

```python
def fitness(tree, x):
```

## `crossover(parent1, parent2)`

The crossover function performs crossover between two parent trees, creating two child trees. It randomly selects nodes from each parent and swaps their values and children, aiming to explore new combinations.

```python
def crossover(parent1, parent2):
```
## `crossover2(parent1, parent2)`

The crossover2 function is an extended version of the basic crossover, addressing specific cases such as swapping variables, numbers, and one-way operators differently. It aims to improve compatibility during the crossover.

```python
def crossover2(parent1, parent2):
```
## `crossover3(parent1, parent2)`

The crossover3 function involves a more complex crossover operation, swapping entire subtrees between selected nodes. It utilizes the replace_subtree helper function.

```python
def crossover3(parent1, parent2):
```

## `mutation(parent)`

The mutation function introduces random changes to a parent tree, promoting diversity in the population. It may change variables to constants or vice versa, and modify operators.

```python
def mutation(parent):
```
## `evaluate_tree(node, x)`

The evaluate_tree function computes the output of an expression tree for a given input x. It recursively evaluates the tree, considering variables, numbers, and various mathematical operations.

```python
def evaluate_tree(node, x):
```
## `simplify_formula(node)`

The simplify_formula function simplifies the mathematical formula represented by an expression tree. It utilizes recursive simplification rules for variables, numbers, and mathematical operations.

```python
def simplify_formula(node):
```
# Using `while` vs `for` Loops in Python

Understanding when to use a `while` loop vs a `for` loop can help you write clearer, more efficient code. Here are examples and explanations for both.

## `for` Loop

Use a `for` loop when you know ahead of time how many times you want to iterate, or when iterating over a sequence (like a list, tuple, or range).

### Example 1: Iterating over a range

```python
for i in range(5):
    print(i)
```

*Use case: You want to run a block of code 5 times.*

### Example 2: Iterating over a list

```python
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

*Use case: You want to perform an action for each item in a list.*

## `while` Loop

Below are scenarios where `while` loops shine due to their flexible exit conditions.

Use a `while` loop when the number of iterations is not known beforehand and the loop should continue based on a condition.

### Example 1: Real-world condition-based loop (monitoring server status)

```python
import time

def check_server():
    # Simulate checking server status
    from random import choice
    return choice(["online", "offline"])

status = check_server()
while status != "online":
    print("Server is offline. Retrying in 5 seconds...")
    time.sleep(5)
    status = check_server()

print("Server is online!")
```

*Use case: You want to keep checking until an external condition changes (like a server coming online). You don't know how many attempts it will take.*

### Example 2: Waiting for user input

```python
user_input = ""
while user_input != "exit":
    user_input = input("Type 'exit' to quit: ")
```

*Use case: You want to keep prompting the user until they type a specific word.*

### Example 3: Simulation until a goal is reached

```python
import random

# Simulate rolling dice until we roll five sixes
six_count = 0
rolls = 0

while six_count < 5:
    roll = random.randint(1, 6)
    rolls += 1
    if roll == 6:
        six_count += 1
    print(f"Roll {rolls}: {roll} (Total sixes: {six_count})")

print(f"It took {rolls} rolls to get five sixes.")
```

*Use case: Simulating a probabilistic process where the number of steps is unpredictable.*

## Summary

| Use Case                           | Use `for` Loop | Use `while` Loop |
| ---------------------------------- | -------------- | ---------------- |
| Known number of iterations         | ✅              |                  |
| Iterating over a sequence          | ✅              |                  |
| Indeterminate number of iterations |                | ✅                |
| Loop controlled by a condition     |                | ✅                |

Use the loop that makes your code more readable and matches the intent of your logic.

## Extra Tip

If you're writing a `while` loop that could easily be rewritten as a `for` loop (like looping through a list or counting to 10), it's usually better to use a `for` loop for clarity. Use `while` when you're dealing with unknowns or external signals like user input, server responses, sensor data, etc.

## Notes

For loops are much more common. In data science, `while` loops are not very frequent—usually you need loops for iterating over datasets, applying transformations, training models over multiple epochs, or processing items in a batch. These tasks have a known structure or length, making `for` loops a natural fit.

That said, **looping through rows in a DataFrame with a `for` loop is not efficient**. It can be done using methods like `.iterrows()` or `.itertuples()`, but this is slow and should generally be avoided for large datasets. Instead, prefer **vectorized operations** using pandas or NumPy whenever possible—they’re significantly faster and more scalable.

`while` loops may show up when you need to poll a service, wait for convergence in optimization, or interact with a process where the stopping condition isn't based on a fixed number of steps.

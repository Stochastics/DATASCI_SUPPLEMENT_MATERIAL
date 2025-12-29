**What is a Data Structure?***
A data structure is just a way to organize and store data so you can use it efficiently.
Think of it like different types of boxes for different things:
If you have a list of names → List
If you have key-value pairs like a name and phone number → Dictionary
If you need only unique items → Set
If you want a bunch of items that never change → Tuple
Different structures are better for different jobs depending on what you need to do — like searching fast, adding/removing easily, or keeping things in order.**


# Common Python Data Structures

| Data Structure | Ordered | Mutable | Allows Duplicates | Unique Keys/Values |
|----------------|---------|---------|-------------------|--------------------|
| List           | ✅      | ✅      | ✅                | ❌                |
| Tuple          | ✅      | ❌      | ✅                | ❌                |
| Set            | ❌      | ✅      | ❌                | ✅ (only unique elements) |
| Dictionary     | ✅      | ✅      | Keys unique       | ✅                |
| String         | ✅      | ❌      | ✅                | ❌                |

---
 
 - A **method** is a function tied to an object, like `A.sort()` for a list, called with a dot to do something with that object.
 
## 1. List
**Create:**
```python
my_list = [1, 2, 3, 4]
```

**Manipulate:**
```python
My_list.append(5)   # Add element to the end
my_list.insert(1, 15)  # Insert element at a specific position
my_list.remove(2)   # Remove element by value
removed_item = my_list.pop()  # Remove last element and return it
my_list[0] = 10     # Update element by index
my_list.sort()      # Sort the list in ascending order
my_list.reverse()   # Reverse the list
```

---

## 2. Tuple
**Create:**
```python
my_tuple = (1, 2, 3)
```

**Manipulate:**
```python
# Tuples are immutable. To change, convert to a list
temp_list = list(my_tuple)
temp_list.append(4)
my_tuple = tuple(temp_list)
```

---

## 3. Set
**Create:**
```python
my_set = {1, 2, 3}
```

**Manipulate:**
```python
my_set.add(4)       # Add element
my_set.remove(2)    # Remove element
```

---

## 4. Dictionary (dict)
**Create:**
```python
my_dict = {"name": "Alice", "age": 25}
```

**Manipulate:**
```python
my_dict["age"] = 26         # Update value
my_dict["city"] = "Paris"   # Add new key-value pair
del my_dict["name"]         # Delete a key

value = my_dict.get("city")    # Get a value safely
keys = list(my_dict.keys())    # Get all keys
values = list(my_dict.values())  # Get all values
items = list(my_dict.items())    # Get all key-value pairs

"city" in my_dict    # Check if key exists (returns True)
```

---

## 5. String
**Create:**
```python
my_string = "Hello, World!"  # Define the string first
```

**Manipulate:**
```python
upper_string = my_string.upper()    # Convert to uppercase
split_string = my_string.split(",") # Split into a list

# Using join() to concatenate multiple strings
words = ["Python", "is", "awesome"]
joined_string = " ".join(words)  # Join list elements with a space
print(joined_string)  # Output: Python is awesome

# String formatting using f-strings
name = "Alice"
age = 30
formatted_string = f"My name is {name} and I am {age} years old."
print(formatted_string)  # Output: My name is Alice and I am 30 years old.

# String slicing
my_string = "Hello, World!"
sliced_string = my_string[0:5]  # Get the first 5 characters
print(sliced_string)  # Output: Hello

# Using replace() to replace a substring
new_string = my_string.replace("World", "Python")  # Replace "World" with "Python"
print(new_string)  # Output: Hello, Python!

# Changing the case of a string
upper_case = my_string.upper()  # Convert to uppercase
lower_case = my_string.lower()  # Convert to lowercase
print(upper_case)  # Output: HELLO, WORLD!
print(lower_case)  # Output: hello, world!
```


- A **method** is a function tied to an object, like `A.sort()` for a list, called with a dot to do something with that object. It’s a member of the object’s class, meaning it’s built into the object’s type (e.g., lists, strings) and designed to work with that specific object. For example, calling `A.sort()` directly modifies the list `A` because the method belongs to it. This makes code cleaner and more intuitive than using standalone functions.




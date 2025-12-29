
# ✅ SQL Cheat Sheet

## 📦 SELECT Basics
```sql
SELECT column1, column2
FROM table_name
WHERE condition
ORDER BY column1 ASC|DESC
LIMIT n
```

## 🔍 Filtering
```sql
-- Exact match
WHERE column = 'value'

-- Pattern matching
WHERE column LIKE 'A%'

-- Range
WHERE column BETWEEN 10 AND 100

-- Null check
WHERE column IS NULL
```

## 🤝 JOINs
```sql
-- Inner Join
SELECT a.col, b.col
FROM table_a a
JOIN table_b b ON a.id = b.id

-- Left Join
SELECT *
FROM table_a
LEFT JOIN table_b ON table_a.id = table_b.id
```

## 📊 GROUP BY + Aggregation
```sql
SELECT department, COUNT(*) AS num_employees, AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING COUNT(*) > 5
```

## 🪟 Window Functions
```sql
-- Row numbering
SELECT name, salary,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num
FROM employees

-- Running total
SELECT order_id, customer_id, amount,
       SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS running_total
FROM orders

-- Rank and dense rank
SELECT name, score,
       RANK() OVER (ORDER BY score DESC) AS rank,
       DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM leaderboard
```

## 📐 CASE Statements
```sql
SELECT name, salary,
       CASE
         WHEN salary > 100000 THEN 'High'
         WHEN salary > 50000 THEN 'Medium'
         ELSE 'Low'
       END AS salary_band
FROM employees
```

## 🔄 CTEs (Common Table Expressions)

### What is a CTE and Why Use It?
CTEs (Common Table Expressions) are temporary result sets that improve readability, reusability, and modularity of SQL queries. They can also support recursion.

### Basic Syntax
```sql
WITH cte_name AS (
    SELECT ...
    FROM ...
    WHERE ...
)
SELECT * FROM cte_name;
```

### Example: Filtering Before Aggregation
```sql
-- Without CTE
SELECT department, COUNT(*) 
FROM (
    SELECT * FROM employees WHERE salary > 100000
) AS high_earners
GROUP BY department;

-- With CTE
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 100000
)
SELECT department, COUNT(*)
FROM high_earners
GROUP BY department;
```

### Recursive CTE Example: Organization Chart
```sql
WITH RECURSIVE org_chart AS (
    SELECT employee_id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    SELECT e.employee_id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.employee_id
)
SELECT * FROM org_chart
ORDER BY level, name;
```

## ⚙️ Stored Procedures

Stored procedures are saved SQL code that can be reused. They're useful for encapsulating logic, reducing repetition, and improving performance.

### Creating a Stored Procedure (MySQL syntax)
```sql
DELIMITER //

CREATE PROCEDURE GetHighEarners(IN min_salary DECIMAL(10,2))
BEGIN
    SELECT * FROM employees
    WHERE salary > min_salary;
END //

DELIMITER ;
```

### Executing a Stored Procedure
```sql
CALL GetHighEarners(100000);
```

### Alter or Drop a Procedure
```sql
DROP PROCEDURE IF EXISTS GetHighEarners;
```

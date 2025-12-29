
# 🧪 Edge Cases in Data Processing: Cheat Sheet

Edge cases are scenarios that occur at the boundaries or extremes of input data, often causing unexpected behavior. Handling them is critical for robust code and reliable results. Below are common edge cases, with explanations, examples, and strategies.

---

## 1. Sentence Splitting with Abbreviations

**Description**: Abbreviations like "Inc." or "Dr." can cause incorrect sentence splitting.

**Example**:
```text
"I work at Acme Inc. I love my job."
```

**Naive Split**:
```python
["I work at Acme Inc", " I love my job"]
```

**Challenges**:
- Periods in abbreviations.
- Multiple punctuation marks (`...`).
- Single-letter initials (e.g., "A. Smith").

**Strategies**:
- List of common abbreviations.
- Regex: `r'(?<!\b(?:Inc|Dr|Mr))\.\s+'`
- NLP tools: `nltk.sent_tokenize`, `spacy`
- Lookahead: Ensure punctuation is followed by a capital letter.

---

## 2. Empty or Null Inputs

**Description**: Null or empty values cause logic errors or crashes.

**Example**:
```python
["apple", "", "banana", None]
```

**Strategies**:
- Validate inputs early.
- Use defaults (`"" → 'N/A'`).
- Pre-filter: `filter(None, data)`
- Unit tests for empty/null input.

---

## 3. Extreme Values and Data Bounds

**Description**: Outliers or extreme values can break logic or cause overflows.

**Example**:
```python
[1, 2, 999999999999999]
```

**Challenges**:
- Overflows (e.g., integer limit).
- Precision errors (e.g., float).
- Out-of-range values.

**Strategies**:
- Type safety (e.g., `Decimal`).
- Boundary validation (`if x > MAX_VALUE`)
- Normalize/extreme value capping.
- Use robust numeric libraries.

---

## 4. Inconsistent Formatting

**Description**: Differences in case, spacing, or encoding cause mismatches.

**Example**:
```python
["New York", "new york ", "NEWYORK"]
```

**Strategies**:
- Normalize: `.lower().strip()`
- Regex cleanups: `re.sub(r'\s+', ' ', text)`
- Fuzzy matching (`fuzzywuzzy`)
- Ensure consistent encoding (`.encode('utf-8')`)

---

## 5. Malformed or Corrupted Data

**Description**: Broken records, wrong data types, or unreadable entries.

**Example**:
```csv
name,age
Alice,25
Bob,invalid
Charlie,30
```

**Strategies**:
- Try/except for parsing.
- Schema validation.
- Log errors.
- Replace with fallback defaults.

--

## 🧠 General Strategies

- **Iterate + Debug**: Use logs and print statements for edge debugging.
- **Test Extensively**: Unit tests for edge inputs.
- **Validate Early**: Catch problems at the input stage.
- **Use Robust Libraries**: e.g., `pandas`, `nltk`, `decimal`.
- **Document Assumptions**: State input/output expectations.

---

## 🗣️ Discussion Prompts

- Why do edge cases matter in production systems?
- How can you systematically discover edge cases?
---

## 🔬 Suggested Class Exercises

| Topic               | Prompt |
|--------------------|--------|
| Sentence splitting | Write a function that tokenizes sentences from text like "Dr. Smith works at Acme Inc. He is happy." |
| Empty values       | Compute average word length from `["cat", "", None, "dog"]`. |
| Extreme values     | Compute mean on `[1, -9999999999, 10**20]`. |
| Formatting issues  | Deduplicate `["John ", "john", "JOHN"]` ignoring case and space. |
| Malformed CSV      | Parse `["Alice,90", "Bob,abc", "Charlie,85"]`, skipping bad rows. |
| Boundary indexes   | Safely access first and last values in a list or dataframe. |

---

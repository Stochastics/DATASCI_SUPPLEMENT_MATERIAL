# 📘 Simple Git Tutorial

## 🚀 Everyday Git Workflow

### ✅ 1. Clone a Repository
```bash
git clone https://github.com/username/repo-name.git
```

### 🌿 2. Create a New Branch
```bash
git checkout -b my-branch-name
```

### 📂 3. See Which Files Changed
```bash
git status
```

### ➕ 4. Add Files to Staging
```bash
git add file1.py file2.txt
# or add everything:
git add .
```

### 📝 5. Commit Changes
```bash
git commit -m "your message here"
```

### 🚀 6. Push Your Changes
```bash
git push
```

---

## 🧰 Miscellaneous (Nice to Know)

### 📃 List All Tracked Files
```bash
git ls-files
```

### 🔄 Fetch Changes from Remote (without merging)
```bash
git fetch
```

---

**That’s it — simple and useful Git for real work.**


---

## 📎 Extra: What's the Difference Between `git fetch` and `git pull`?

- `git fetch`: Downloads updates from the remote, but does **not** change your current files or branches. You can review changes before applying them.
- `git pull`: Does **both** a `fetch` and a `merge`. It updates your current branch with any changes from the remote branch.

### 📥 Pull Remote Changes and Merge
```bash
git pull
```


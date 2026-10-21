# Contribution Guidelines - AUDIOARTIFACT

Welcome to the **AUDIOARTIFACT** project! To maintain code stability and prevent accidental bugs in the production pipeline, please follow these guidelines.

---

## 🚫 Rules for Branches
- **Do NOT push directly to `main` branch.**
- All changes must be pushed to dedicated **feature / fix branches**.
- Changes are merged into `main` via **Pull Requests (PR)** with review.

---

## 🛠️ Step-by-Step Contribution Workflow

### 1. Sync with the Latest Code
Before starting any new work, always pull the latest changes from `main`:
```bash
git checkout main
git pull origin main
```

### 2. Create a New Branch
Branch names should be descriptive of the task:
```bash
# For a new feature:
git checkout -b feature/your-feature-name

# For bug fixes:
git checkout -b fix/issue-description
```

### 3. Work on Your Changes & Test
- Make your code changes.
- Ensure existing pipelines or tests don't break.
- Do not commit large binary/cache files, virtual environments, or sensitive keys.

### 4. Commit and Push
```bash
git add .
git commit -m "feat: brief description of what was changed"
git push -u origin feature/your-feature-name
```

### 5. Open a Pull Request (PR)
1. Go to the GitHub repository: [AUDIOARTIFACT Repository](https://github.com/tripathi0704/AUDIOARTIFACT)
2. You will see a banner with **"Compare & pull request"**. Click on it.
3. Write a clear description of:
   - What changes were made?
   - How to test or run them?
4. Submit the Pull Request for review.
5. Once reviewed and approved, it will be merged into `main`.

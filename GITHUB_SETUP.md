# GitHub Repository Setup Guide

## Step 1: Create Repository on GitHub

1. **Go to GitHub**: Visit [github.com](https://github.com) and sign in (or create an account)

2. **Create New Repository**:
   - Click the "+" icon in the top right corner
   - Select "New repository"
   - Repository name: `collatz-conjecture-ml` (or any name you prefer)
   - Description: "Neural network models to predict Collatz conjecture sequences"
   - Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository on GitHub, you'll see a page with setup instructions. Use these commands:

```bash
# Make sure you're in the project directory
cd "/Users/minta/Christian Jin (During the washington trip)"

# Add all files (if not already done)
git add .

# Create your first commit
git commit -m "Initial commit: Collatz conjecture ML predictor with multiple model architectures"

# Add the GitHub repository as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Verify

1. Go to your GitHub repository page
2. You should see all your files there
3. The README.md will be displayed on the main page

## Quick Reference Commands

```bash
# Check status
git status

# Add files
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push

# Pull latest changes
git pull
```

## What's Included in .gitignore

The `.gitignore` file I created excludes:
- Virtual environment (`venv/`)
- Python cache files (`__pycache__/`)
- Model checkpoints (`models/*.pth`)
- IDE files
- OS files (`.DS_Store`)

This keeps your repository clean and prevents uploading large/unnecessary files.


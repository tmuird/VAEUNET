#!/bin/bash

# Save current branch name
CURRENT_BRANCH=$(git branch --show-current)
TARGET_BRANCH=$1

if [ -z "$TARGET_BRANCH" ]; then
  echo "Usage: ./switch_branch.sh <branch-name>"
  exit 1
fi

echo "Switching from $CURRENT_BRANCH to $TARGET_BRANCH"

# Clean untracked files to prevent conflicts
git clean -f -d

# Stash any uncommitted changes
git stash save "Auto-stash before switching to $TARGET_BRANCH"

# Switch branch
git checkout $TARGET_BRANCH

# Set up environment based on branch
if [ "$TARGET_BRANCH" == "sagemaker" ]; then
  echo "Setting up SageMaker environment..."
  # Install SageMaker dependencies if needed
  pip install -q boto3 sagemaker
elif [ "$TARGET_BRANCH" == "main" ]; then
  echo "Setting up local development environment..."
  # Any setup needed for local development
fi

echo "Successfully switched to $TARGET_BRANCH branch"

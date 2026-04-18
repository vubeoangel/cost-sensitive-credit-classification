#!/usr/bin/env bash
# One-shot publish script. Run from the project root:
#   bash publish_to_github.sh
#
# Creates a public GitHub repo named "cost-sensitive-credit-classification"
# and pushes the existing local main branch.
#
# Requires: gh CLI (https://cli.github.com — install with `brew install gh`)

set -e

REPO_NAME="cost-sensitive-credit-classification"
VISIBILITY="--public"

cd "$(dirname "$0")"

# ─── Preflight ────────────────────────────────────────────────────────
if ! command -v gh >/dev/null 2>&1; then
  echo "❌ GitHub CLI (gh) is not installed."
  echo
  echo "Install it with:  brew install gh"
  echo "Then rerun:       bash publish_to_github.sh"
  echo
  echo "─── or publish manually ───────────────────────────────────────"
  echo "1. Open https://github.com/new"
  echo "2. Name the repo: $REPO_NAME (Public, no README/gitignore/license)"
  echo "3. Run these commands:"
  echo "     git remote add origin https://github.com/<your-username>/$REPO_NAME.git"
  echo "     git push -u origin main"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "🔑 You're not signed in to gh. Running 'gh auth login'…"
  gh auth login
fi

# ─── Create + push ────────────────────────────────────────────────────
echo "🚀 Creating $REPO_NAME and pushing main branch…"
gh repo create "$REPO_NAME" $VISIBILITY --source=. --remote=origin --push

echo
echo "✅ Done. Your repo is live:"
gh repo view --web >/dev/null 2>&1 &
gh repo view --json url --jq .url

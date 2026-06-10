---
title: "Initial Tools Setup"
author: "Sae-Hwan Park"
date: 2026-06-10
description: "Walkthrough guide for initial tooling setup (git, ssh key, github, uv)"
---

# Check if Git is installed

```bash
git --version
```

If not installed, install it using your package manager (e.g. `apt`, `brew`, `dnf`).

```bash
# Ubuntu/Debian
sudo apt install git
```

```bash
# Fedora
sudo dnf install git
```

```bash
# Homebrew (macOS)
brew install git
```

# Add SSH key and link to GitHub

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
```

```bash
# Add SSH key to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

Then, open your browser and add the SSH key to your GitHub account.

- Navigate to [GitHub Settings](https://github.com/settings/keys) and click "New SSH key"
- Paste the contents of `~/.ssh/id_ed25519.pub` into the "Key" field
- Give the key a descriptive title

Also you may need to setup your account name and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

# Install uv

Now let's install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Check the installation:

```bash
uv --version
```

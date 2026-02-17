# Contributing to Kalshi Arbitrage Bot

Thank you for your interest in contributing! This document outlines the process for contributing to this project.

## Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (optional)
- Git

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/arbitrage_bot.git
   cd arbitrage_bot
   ```

3. Set up development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. Create a branch for your changes:
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

## Development Workflow

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Run `ruff check src/ tests/` before committing
- Run `mypy src/` to check types

### Testing

- Write tests for new functionality
- Ensure all tests pass: `pytest tests/ -v`
- Aim for meaningful test coverage

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(arbitrage): add cross-market opportunity detection

Implemented detection of price differences between related markets.
Added confidence scoring based on liquidity and fill probability.

Closes #123
```

### Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of your changes
4. Request review from maintainers

## Code Structure

```
src/
├── clients/       # API clients
├── core/          # Business logic
├── execution/     # Order execution
├── monitoring/    # Metrics & logging
├── api/           # REST API
└── utils/         # Utilities
```

## Adding New Features

1. Consider the impact on existing modules
2. Add appropriate tests
3. Update configuration if needed
4. Document in README if user-facing

## Reporting Issues

- Use GitHub Issues for bugs
- Provide clear reproduction steps
- Include relevant logs and configuration

## Questions?

Open an issue for discussion or reach out via Discord.

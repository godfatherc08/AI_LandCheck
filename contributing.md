# Contributing to Veritas

We love your input! We want to make contributing as easy and transparent as possible.

## Development Process

1. Fork the repo and create your branch from `main`
2. Install development dependencies: `pip install -r requirements.txt`
3. Make your changes
4. Run tests
6. Submit a pull request

## How to Contribute

### Report Bugs
- Use GitHub Issues
- Describe exact steps to reproduce
- Include version, OS, and relevant logs

### Suggest Features
- Use GitHub Issues with label `enhancement`
- Explain the feature and its value

### Code Contributions
- Follow PEP 8 style guide
- Add docstrings to new functions
- Write tests for new functionality
- Update documentation

## Adding New Verification Tools

1. Create new tool in `verification_engine/agent/offline_tools.py`
2. Add to `OFFLINE_TOOLS` registry
3. Update `verifier.py` to include your tool
4. Add tests
5. Update documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **DO NOT** open a public issue.
2. Email **gaurav14cs17@gmail.com** with:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive an acknowledgment within 48 hours.
4. We aim to release a fix within 7 days of confirmation.

## Security Best Practices

- Never commit API keys, tokens, or credentials.
- Use environment variables for sensitive configuration.
- Keep dependencies updated (`pip install --upgrade`).
- Review third-party model weights before loading (`torch.load` with `weights_only=True`).

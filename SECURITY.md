# Security Guidelines

This document outlines security best practices for the IBKR Trading Application to protect sensitive information and ensure safe trading operations.

## Environment Variables Security

### Required Environment Variables

The application requires sensitive information to be stored as environment variables instead of configuration files:

- `IBKR_ACCOUNT_ID` (Required): Your Interactive Brokers account ID
- `IBKR_HOST` (Optional): Override IBKR connection host
- `IBKR_PORT` (Optional): Override IBKR connection port
- `IBKR_CLIENT_ID` (Optional): Override IBKR client ID
- `IBKR_PAPER_TRADING` (Optional): Override paper trading mode

### Environment Variable Best Practices

#### 1. Use .env Files for Development
```bash
# Copy the example file
cp .env.example .env

# Edit with your actual values
IBKR_ACCOUNT_ID=DU123456
IBKR_PAPER_TRADING=true
```

#### 2. Never Commit Sensitive Data
- `.env` files are already added to `.gitignore`
- Never commit account IDs, API keys, or passwords
- Use separate configurations for dev/staging/production

#### 3. File Permissions
Restrict access to your environment files:
```bash
chmod 600 .env  # Only owner can read/write
```

#### 4. Production Deployment
For production systems, use:
- System environment variables
- Container orchestration secrets (Kubernetes secrets, Docker secrets)
- Cloud provider secret management (AWS Secrets Manager, Azure Key Vault, etc.)

## Trading Security

### Paper Trading vs Live Trading

#### Always Start with Paper Trading
```bash
export IBKR_ACCOUNT_ID=DU123456  # Paper account
export IBKR_PAPER_TRADING=true
```

#### Live Trading Precautions
When ready for live trading:
```bash
export IBKR_ACCOUNT_ID=U1234567   # Live account
export IBKR_PAPER_TRADING=false
```

**⚠️ WARNING: Live trading involves real money and substantial risk of loss**

### Risk Management Settings

Review and adjust risk parameters in `config.toml`:
```toml
[risk]
max_daily_loss_pct = 0.02      # 2% maximum daily loss
max_position_size_pct = 0.15   # 15% maximum position size
stop_loss_pct = 0.05           # 5% stop loss
take_profit_pct = 0.10         # 10% take profit
```

## Network Security

### IBKR Connection Security

1. **Use Secure Connections**: IBKR TWS/Gateway use encrypted connections
2. **Firewall Rules**: Restrict access to IBKR ports (7496/7497) to localhost only
3. **VPN**: Consider using VPN for additional security when trading remotely

### API Security

1. **Client ID Management**: Use unique client IDs for different applications
2. **Connection Monitoring**: Monitor for unauthorized connection attempts
3. **Session Management**: Implement proper session timeouts

## Data Protection

### Sensitive Data Handling

1. **Logging**: Ensure account IDs and sensitive data are not logged
2. **Memory**: Clear sensitive data from memory when possible
3. **Storage**: Never store credentials in plain text files

### Backup Security

1. **Configuration Backups**: Only backup non-sensitive configuration files
2. **Trade Data**: Encrypt any stored trade history or performance data
3. **Recovery**: Have secure recovery procedures for lost credentials

## Development Security

### Code Security

1. **Dependencies**: Regularly update dependencies for security patches
2. **Code Review**: Review code for hardcoded credentials or sensitive data
3. **Static Analysis**: Use tools to scan for security vulnerabilities

### Testing Security

1. **Test Data**: Use fake account IDs and data for tests
2. **Environment Isolation**: Keep test and production environments separate
3. **CI/CD**: Ensure build systems don't expose sensitive data

## Incident Response

### If Credentials Are Compromised

1. **Immediate Actions**:
   - Change IBKR account passwords immediately
   - Revoke API access if available
   - Monitor account for unauthorized activity

2. **Investigation**:
   - Review application logs for suspicious activity
   - Check for unauthorized trades or access
   - Document the incident

3. **Recovery**:
   - Generate new credentials
   - Update environment variables
   - Restart application with new credentials

### Monitoring

1. **Account Monitoring**: Regularly check IBKR account for unexpected activity
2. **Application Monitoring**: Monitor application logs for errors or unusual behavior
3. **Performance Monitoring**: Watch for unexpected trading patterns

## Compliance

### Record Keeping

1. **Trade Records**: Maintain secure records of all trades
2. **Configuration Changes**: Log changes to trading parameters
3. **Access Logs**: Keep logs of who accessed the system when

### Regulatory Compliance

1. **Know Your Obligations**: Understand trading regulations in your jurisdiction
2. **Risk Disclosure**: Ensure you understand the risks of automated trading
3. **Documentation**: Maintain proper documentation of your trading strategy

## Emergency Procedures

### Emergency Shutdown

In case of malfunction or risk:

1. **Stop the Application**:
   ```bash
   # Send interrupt signal
   Ctrl+C
   
   # Or kill the process
   pkill -f first-trading-app
   ```

2. **Manual Intervention**:
   - Log into IBKR TWS/Gateway manually
   - Cancel any pending orders
   - Close positions if necessary

3. **Disconnect from IBKR**:
   - Stop TWS/Gateway application
   - Verify no active connections

### Contact Information

Keep emergency contact information readily available:
- IBKR Customer Service: [Your region's number]
- Your broker representative
- Technical support contacts

## Security Checklist

Before going live with real money:

- [ ] Environment variables are properly configured
- [ ] `.env` file has restricted permissions (600)
- [ ] No sensitive data in source control
- [ ] Risk limits are appropriately set
- [ ] Paper trading has been thoroughly tested
- [ ] Emergency procedures are documented and understood
- [ ] IBKR account security is enabled (2FA, etc.)
- [ ] Regular monitoring procedures are in place
- [ ] Backup and recovery procedures are tested

## Additional Resources

- [IBKR Security Center](https://www.interactivebrokers.com/en/index.php?f=2334)
- [TWS API Security](https://interactivebrokers.github.io/tws-api/)
- [Trading Risk Disclosure](https://www.interactivebrokers.com/en/index.php?f=971)

---

**Remember: Trading involves substantial risk of loss. Never trade with money you cannot afford to lose.**
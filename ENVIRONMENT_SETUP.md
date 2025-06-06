# Environment Variable Setup

This trading application uses environment variables to store sensitive configuration data such as IBKR account credentials. This approach keeps sensitive information out of configuration files and source code.

## Required Environment Variables

### IBKR_ACCOUNT_ID (Required)
Your Interactive Brokers account ID.

**Example:**
```bash
export IBKR_ACCOUNT_ID=DU123456
```

For paper trading accounts, this typically starts with "DU" followed by numbers.
For live trading accounts, this is your actual account number.

## Optional Environment Variables

### IBKR_HOST
Override the default IBKR TWS/Gateway host.

**Default:** `127.0.0.1`

**Example:**
```bash
export IBKR_HOST=192.168.1.100
```

### IBKR_PORT
Override the default IBKR TWS/Gateway port.

**Defaults:**
- Paper Trading: `7497`
- Live Trading: `7496`

**Example:**
```bash
export IBKR_PORT=7497
```

### IBKR_CLIENT_ID
Override the default client ID for IBKR connection.

**Default:** `1`

**Example:**
```bash
export IBKR_CLIENT_ID=2
```

### IBKR_PAPER_TRADING
Override the paper trading mode setting.

**Default:** `true`

**Example:**
```bash
export IBKR_PAPER_TRADING=false  # Enable live trading
```

## Setup Methods

### Method 1: Using .env File (Recommended)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your actual values:
   ```bash
   # IBKR Configuration Environment Variables
   IBKR_ACCOUNT_ID=DU123456
   
   # Optional overrides
   # IBKR_HOST=127.0.0.1
   # IBKR_PORT=7497
   # IBKR_CLIENT_ID=1
   # IBKR_PAPER_TRADING=true
   ```

3. The application will automatically load these variables from the `.env` file at startup.

### Method 2: System Environment Variables

Set the environment variables in your shell:

```bash
# For bash/zsh
export IBKR_ACCOUNT_ID=DU123456
export IBKR_PAPER_TRADING=true

# Make permanent by adding to ~/.bashrc or ~/.zshrc
echo 'export IBKR_ACCOUNT_ID=DU123456' >> ~/.bashrc
```

```powershell
# For Windows PowerShell
$env:IBKR_ACCOUNT_ID="DU123456"
$env:IBKR_PAPER_TRADING="true"

# Make permanent using System Properties or:
[Environment]::SetEnvironmentVariable("IBKR_ACCOUNT_ID", "DU123456", "User")
```

### Method 3: Docker Environment

When running in Docker, pass environment variables:

```bash
docker run -e IBKR_ACCOUNT_ID=DU123456 -e IBKR_PAPER_TRADING=true your-trading-app
```

Or use an environment file:

```bash
docker run --env-file .env your-trading-app
```

## Security Best Practices

### 🔒 Never Commit Sensitive Data
- Add `.env` to your `.gitignore` file (already done)
- Never commit account IDs, API keys, or passwords to version control
- Use separate `.env` files for different environments (dev, staging, prod)

### 🔒 File Permissions
Restrict access to your `.env` file:

```bash
chmod 600 .env  # Only owner can read/write
```

### 🔒 Environment Separation
Use different account IDs for different environments:

```bash
# Development/Testing
IBKR_ACCOUNT_ID=DU123456  # Paper trading account

# Production
IBKR_ACCOUNT_ID=U1234567  # Live trading account (be very careful!)
```

## Verification

To verify your environment variables are set correctly:

```bash
# Check if required variable is set
echo $IBKR_ACCOUNT_ID

# Or check all IBKR variables
env | grep IBKR
```

The application will display an error message at startup if required environment variables are missing.

## Troubleshooting

### Error: "IBKR_ACCOUNT_ID environment variable is required"

**Solution:** Set the `IBKR_ACCOUNT_ID` environment variable using one of the methods above.

### Error: "Invalid IBKR_PORT value"

**Solution:** Ensure `IBKR_PORT` is set to a valid integer (typically 7496 or 7497).

### Error: "Invalid IBKR_CLIENT_ID value"

**Solution:** Ensure `IBKR_CLIENT_ID` is set to a valid integer.

### Environment Variables Not Loading

1. Check that `.env` file exists in the application root directory
2. Verify file permissions allow reading
3. Ensure no syntax errors in `.env` file (no spaces around `=`)
4. Restart the application after making changes

## Integration with IBKR TWS/Gateway

Make sure your IBKR TWS or Gateway is configured to accept connections:

1. **TWS Configuration:**
   - Go to File → Global Configuration → API → Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Set "Socket port" to match your `IBKR_PORT` (default: 7497 for paper)
   - Add your `IBKR_CLIENT_ID` to "Trusted IPs" if needed

2. **Gateway Configuration:**
   - Similar settings available in Gateway interface
   - Ensure the account ID matches your `IBKR_ACCOUNT_ID`

3. **Paper Trading vs Live Trading:**
   - Paper trading typically uses port 7497
   - Live trading typically uses port 7496
   - Ensure `IBKR_PAPER_TRADING` matches your intended mode

## Example Complete Setup

1. Create `.env` file:
   ```bash
   IBKR_ACCOUNT_ID=DU123456
   IBKR_PAPER_TRADING=true
   ```

2. Start IBKR TWS/Gateway with paper trading enabled

3. Run the application:
   ```bash
   cargo run
   ```

4. Verify in logs:
   ```
   INFO Configuration loaded successfully
   INFO Configuration Summary:
   INFO   IBKR Host: 127.0.0.1:7497
   INFO   Paper Trading: true
   ```

Your application should now connect successfully to IBKR using your environment-configured credentials.
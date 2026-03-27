# Setting up GitHub Integration for Modal Deployment

Follow these steps to enable automatic deployment to Modal when you push to GitHub.

## Step 1: Get Your Modal Tokens

1. Go to [Modal Settings](https://modal.com/settings/tokens)
2. Click "Create new token"
3. Give it a name like "GitHub Actions"
4. Copy both the **MODAL_TOKEN_ID** and **MODAL_TOKEN_SECRET**

## Step 2: Add Tokens to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add these two secrets:
   - **MODAL_TOKEN_ID**: Your Modal token ID
   - **MODAL_TOKEN_SECRET**: Your Modal token secret

## Step 3: Deploy Manually (First Time)

Before setting up the GitHub Action, deploy manually to ensure everything works:

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# Deploy
modal deploy modal_app.py
```

## Step 4: Test the GitHub Action

1. Commit and push your changes:
```bash
git add .
git commit -m "Add Modal deployment with GitHub Actions"
git push origin main
```

2. Go to **Actions** tab in your GitHub repo to watch the deployment

## What Happens Next

- ✅ **Automatic Deployment**: Every push to main triggers deployment
- ✅ **Smart Triggers**: Only deploys when Modal-related files change
- ✅ **URL Sharing**: PR comments include the live API URL
- ✅ **Error Logging**: Failed deployments show detailed error logs

## Deployment Workflow

The GitHub Action will:
1. Check out your code
2. Set up Python 3.11
3. Install Modal CLI
4. Deploy `modal_app.py`
5. Share the live URL in PR comments

## Dual Deployment Strategy

You now have both deployments running:
- **Hugging Face**: `app_port: 7860` (fixed)
- **Modal**: Serverless, auto-scaling

This gives you:
- 🛡️ **Redundancy**: If one fails, the other works
- ⚡ **Speed**: Modal deploys much faster
- 💰 **Cost Control**: Modal scales to zero when idle

## Testing the Deployed API

Once deployed, test your Modal API:

```bash
# Health check
curl https://your-app-name.modal.run/health

# Classify audio
curl -X POST https://your-app-name.modal.run/classify \
  -H "Content-Type: application/json" \
  -d '{"audio": "base64_encoded_audio_data"}'
```

## Troubleshooting

### Common Issues:

1. **Token not working**: Ensure tokens are correctly copied to GitHub Secrets
2. **Deployment fails**: Check the Action logs for specific Python errors
3. **App not found**: Make sure the app name "yamnet-sound-classifier" matches in `modal_app.py`

### Debug Commands:

```bash
# Check Modal app status
modal app list

# View app logs
modal app logs yamnet-sound-classifier

# Test locally
modal run modal_app.py
```

## Next Steps

- ✅ Set up Modal tokens
- ✅ Add GitHub secrets  
- ✅ Test manual deployment
- ✅ Push and test GitHub Action
- ✅ Verify both deployments work

Your YAMNet API will now be automatically deployed to both Hugging Face and Modal with every push! 🚀

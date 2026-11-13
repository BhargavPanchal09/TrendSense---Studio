# Connect the live trend sources

The backend uses this private credential:

- `YOUTUBE_API_KEY` to read public `videos.list` entries from YouTube's `mostPopular` chart.

## Google / YouTube key

1. Sign in to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project, or select a project you already use.
3. Open **APIs & Services → Library**, search for **YouTube Data API v3**, and click **Enable**.
4. Open **APIs & Services → Credentials → Create credentials → API key**.
5. In the key's settings, restrict API usage to **YouTube Data API v3**. This app only reads public charts; it does not need access to your YouTube account.
6. Copy the API key into `YOUTUBE_API_KEY` in the backend `.env` file.

Google's official instructions: https://developers.google.com/youtube/registering_an_application . The API key is read by the backend and must not be added to frontend code.

## Reddit

Reddit is disabled in the live dashboard. Reddit's current [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy) requires explicit approval before API access and restricts research use to its Researcher Program. Do not add Reddit credentials or access Reddit data for this project unless the required approval is granted.

## Add the values privately

From the `backend` folder in VS Code's PowerShell terminal:

```powershell
Copy-Item .env.example .env
code .env
```

Replace the blank values with your credentials, save the file, and restart the backend. The root `.gitignore` ignores `.env`; the example file has blank values. Never paste credentials into chat or commit the `.env` file.

After restart, choose **India** in TrendSense and run an analysis. The source status panel should show YouTube and Google search feeds, with the returned item counts. Reddit is intentionally omitted. A working connection improves the evidence sample, but chart membership does not guarantee future video views or model accuracy.

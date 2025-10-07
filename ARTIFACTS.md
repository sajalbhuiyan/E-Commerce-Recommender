# External large artifacts

This project keeps some large model artifacts outside of the Git repository and references them by S3 URLs to avoid inflating the git history.

Please do not commit these files to the repository. Instead, download them from the given URLs when setting up the project.

- svd_model.pkl:
  - S3 URL: https://movie-recommendation-files.s3.us-east-1.amazonaws.com/svd_model.pkl
- rf_recommender.pkl:
  - S3 URL: https://movie-recommendation-files.s3.us-east-1.amazonaws.com/rf_recommender.pkl

How to fetch an artifact locally:

```powershell
Invoke-WebRequest -Uri "<URL>" -OutFile ".\<filename>"
```

Or use the Streamlit sidebar "Download artifact" feature in this app to paste the URL and save the file into the project root.

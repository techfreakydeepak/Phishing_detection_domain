from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

def extract_features_from_url(url):
    # Extract features from the URL
    # This is a placeholder, replace it with actual feature extraction logic
    features = {
        "id": [1],
        "NumDots": [url.count('.')],
        "SubdomainLevel": [url.count('.') - 1],
        "PathLevel": [url.count('/') - 2],
        "UrlLength": [len(url)],
        "NumDash": [url.count('-')],
        "NumDashInHostname": [url.split('/')[2].count('-')],
        "AtSymbol": [url.count('@')],
        "TildeSymbol": [url.count('~')],
        "NumUnderscore": [url.count('_')],
        "NumPercent": [url.count('%')],
        "NumQueryComponents": [url.count('?')],
        "NumAmpersand": [url.count('&')],
        "NumHash": [url.count('#')],
        "NumNumericChars": [sum(c.isdigit() for c in url)],
        "NoHttps": [1 if not url.startswith('https') else 0],
        "RandomString": [0],  # Placeholder, implement actual logic
        "IpAddress": [1 if url.split('/')[2].replace('.', '').isdigit() else 0],
        "DomainInSubdomains": [0],  # Placeholder, implement actual logic
        "DomainInPaths": [0],  # Placeholder, implement actual logic
        "HttpsInHostname": [0],  # Placeholder, implement actual logic
        "HostnameLength": [len(url.split('/')[2])],
        "PathLength": [len(url.split('/')[-1])],
        "QueryLength": [len(url.split('?')[-1]) if '?' in url else 0],
        "DoubleSlashInPath": [url.count('//')],
        "NumSensitiveWords": [0],  # Placeholder, implement actual logic
        "EmbeddedBrandName": [0],  # Placeholder, implement actual logic
        "PctExtHyperlinks": [0],  # Placeholder, implement actual logic
        "PctExtResourceUrls": [0],  # Placeholder, implement actual logic
        "ExtFavicon": [0],  # Placeholder, implement actual logic
        "InsecureForms": [0],  # Placeholder, implement actual logic
        "RelativeFormAction": [0],  # Placeholder, implement actual logic
        "ExtFormAction": [0],  # Placeholder, implement actual logic
        "AbnormalFormAction": [0],  # Placeholder, implement actual logic
        "PctNullSelfRedirectHyperlinks": [0],  # Placeholder, implement actual logic
        "FrequentDomainNameMismatch": [0],  # Placeholder, implement actual logic
        "FakeLinkInStatusBar": [0],  # Placeholder, implement actual logic
        "RightClickDisabled": [0],  # Placeholder, implement actual logic
        "PopUpWindow": [0],  # Placeholder, implement actual logic
        "SubmitInfoToEmail": [0],  # Placeholder, implement actual logic
        "IframeOrFrame": [0],  # Placeholder, implement actual logic
        "MissingTitle": [0],  # Placeholder, implement actual logic
        "ImagesOnlyInForm": [0],  # Placeholder, implement actual logic
        "SubdomainLevelRT": [0],  # Placeholder, implement actual logic
        "UrlLengthRT": [0],  # Placeholder, implement actual logic
        "PctExtResourceUrlsRT": [0],  # Placeholder, implement actual logic
        "AbnormalExtFormActionR": [0],  # Placeholder, implement actual logic
        "ExtMetaScriptLinkRT": [0],  # Placeholder, implement actual logic
        "PctExtNullSelfRedirectHyperlinksRT": [0],  # Placeholder, implement actual logic
    }
    return features

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        url = request.form['url']
        logging.info(f"Received URL: {url}")
        
        features = extract_features_from_url(url)
        pred_df = pd.DataFrame(features)
        logging.info(f"Extracted features: {pred_df}")
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        result_text = "Phishing" if results[0] == 1 else "Legitimate"
        logging.info(f"Prediction result: {result_text}")
        
    except CustomException as e:
        result_text = f"An error occurred: {str(e)}"
        logging.error(result_text)
    
    return render_template('home.html', result=result_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

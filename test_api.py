import requests
import pandas as pd

# The 10 applicants from Table 1
applicants = [
    {
        "Age": 65,
        "Sex": "female",
        "Job": 1,
        "Housing": "free",
        "Saving_accounts": "moderate",
        "Checking_account": 19,
        "Credit_amount": 1405068,
        "Duration": 8,
        "Purpose": "vacation/others",
        "Label": "Bad",
    },
    {
        "Age": 56,
        "Sex": "female",
        "Job": 3,
        "Housing": "rent",
        "Saving_accounts": "moderate",
        "Checking_account": 24,
        "Credit_amount": 4589191,
        "Duration": 47,
        "Purpose": "radio/TV",
        "Label": "Good",
    },
    {
        "Age": 20,
        "Sex": "male",
        "Job": 0,
        "Housing": "free",
        "Saving_accounts": "rich",
        "Checking_account": 29,
        "Credit_amount": 5835642,
        "Duration": 17,
        "Purpose": "furniture/equipment",
        "Label": "Good",
    },
    {
        "Age": 21,
        "Sex": "male",
        "Job": 0,
        "Housing": "rent",
        "Saving_accounts": "quite rich",
        "Checking_account": 24,
        "Credit_amount": 7924132,
        "Duration": 23,
        "Purpose": "vacation/others",
        "Label": "Bad",
    },
    {
        "Age": 55,
        "Sex": "female",
        "Job": 2,
        "Housing": "free",
        "Saving_accounts": "quite rich",
        "Checking_account": 23,
        "Credit_amount": 7840408,
        "Duration": 47,
        "Purpose": "education",
        "Label": "Good",
    },
    {
        "Age": 27,
        "Sex": "male",
        "Job": 3,
        "Housing": "free",
        "Saving_accounts": "little",
        "Checking_account": 8,
        "Credit_amount": 2013894,
        "Duration": 39,
        "Purpose": "repairs",
        "Label": "Good",
    },
    {
        "Age": 33,
        "Sex": "female",
        "Job": 3,
        "Housing": "own",
        "Saving_accounts": "little",
        "Checking_account": 10,
        "Credit_amount": 6049706,
        "Duration": 13,
        "Purpose": "furniture/equipment",
        "Label": "Good",
    },
    {
        "Age": 50,
        "Sex": "male",
        "Job": 0,
        "Housing": "rent",
        "Saving_accounts": "quite rich",
        "Checking_account": 17,
        "Credit_amount": 6958839,
        "Duration": 39,
        "Purpose": "vacation/others",
        "Label": "Good",
    },
    {
        "Age": 27,
        "Sex": "female",
        "Job": 3,
        "Housing": "rent",
        "Saving_accounts": "rich",
        "Checking_account": 15,
        "Credit_amount": 2319540,
        "Duration": 43,
        "Purpose": "business",
        "Label": "Good",
    },
    {
        "Age": 45,
        "Sex": "male",
        "Job": 1,
        "Housing": "rent",
        "Saving_accounts": "quite rich",
        "Checking_account": 8,
        "Credit_amount": 7679375,
        "Duration": 24,
        "Purpose": "furniture/equipment",
        "Label": "Good",
    },
]

url = "http://localhost:8000/predict"
results = []

for i, app in enumerate(applicants):
    # Separate the true label from the payload going to the API
    payload = app.copy()
    true_label = payload.pop("Label")

    try:
        response = requests.post(url, json=payload)
        response_data = response.json()
        prediction = response_data.get("Risk")
        prob = response_data.get("Probability")

        results.append(
            {
                "ID": i + 1,
                "Sex": app["Sex"],
                "True Label": true_label,
                "Prediction": prediction,
                "Prob (Good)": f"{prob:.4f}",
            }
        )
    except Exception as e:
        print(f"Error on applicant {i+1}: {e}")

# Print cleanly for our deliverables
df = pd.DataFrame(results)
print("\n--- Table 2: API Inference Results ---")
print(df.to_markdown(index=False))

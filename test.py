import requests

url = "https://student-performance-predictor.up.railway.app/predict_api"

data = {
 "gender": "male",
 "race_ethnicity": "group A",
 "parental_level_of_education": "bachelor's degree",
 "lunch": "standard",
 "test_preparation_course": "none",
 "reading_score": 72,
 "writing_score": 74
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.text)
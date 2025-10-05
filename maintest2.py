import requests

# List of sample user profiles
test_profiles = [
    {
        "skills": ["Python", "SQL"],
        "interests": ["Technology"],
        "personality": {"analytical": 0.9, "creative": 0.5, "social": 0.6},
        "education": "Master",
        "experience": 3
    },
    {
        "skills": ["Creative Writing", "UI/UX", "Excel"],
        "interests": ["Design", "Business"],
        "personality": {"analytical": 0.3, "creative": 0.9, "social": 0.7},
        "education": "Bachelor",
        "experience": 2
    },
    {
        "skills": ["Java", "Machine Learning", "Statistics", "SQL"],
        "interests": ["AI", "Data Science"],
        "personality": {"analytical": 0.8, "creative": 0.4, "social": 0.5},
        "education": "PhD",
        "experience": 5
    }
]

# Send POST requests and print results
for idx, profile in enumerate(test_profiles, 1):
    response = requests.post("http://127.0.0.1:8000/predict", json=profile)
    print(f"--- Profile {idx} ---")
    print(response.json())
    print()

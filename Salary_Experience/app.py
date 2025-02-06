from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load dataset
data = pd.read_csv('D:/WebiSoftTech/RANDOM FOREST CLASSIFIER/Salary_Experience/Salary_Experience.csv')

# Prepare data for training
X = data[['YearsExperience']]
y = data['Salary']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    # Get experience and job from the request
    experience = request.json.get('experience')
    job = request.json.get('job')  # Optional key
    
    # Validate input
    if experience is None:
        return jsonify({'error': 'Experience is required'}), 400
    
    # Make prediction
    predicted_salary = model.predict([[experience]])
    
    # Return the predicted salary along with job info if provided
    return jsonify({
        'predicted_salary': predicted_salary[0],
        'job': job  # Include job in the response if provided
    })

if __name__ == '__main__':
    app.run(debug=True)
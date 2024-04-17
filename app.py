from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS, cross_origin
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
CORS(app)

# Initialize Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Mock User class for demonstration purposes
class User(UserMixin):
    pass

@login_manager.user_loader
def load_user(user_id):
    user = User()
    user.id = user_id
    return user

# Load the Colab-based model
filename = 'modelForPrediction.sav'
loaded_model = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open('standardScaler.sav', 'rb'))

# Mock database for storing prediction history
prediction_history_db = []

@app.route('/', methods=['GET'])
@cross_origin()
def home_page():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        # Mock user authentication, replace with your actual authentication logic
        user = User()
        user.id = 1
        login_user(user)
        return redirect(url_for('home_page'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home_page'))

@app.route('/history')
@login_required
def prediction_history():
    return render_template('history.html', history=prediction_history_db)

@app.route('/predict', methods=['POST'])
@cross_origin()
@login_required
def predict():
    try:
        # Read and process the form data
        input_data = [float(value) for value in request.form['data'].split(',')]

        # Scale input data and make the prediction
        scaled_data = scaler.transform([input_data])
        prediction = loaded_model.predict(scaled_data)

        # Interpret the prediction result
        result_message = (
            "You have Parkinson's Disease. Please consult a specialist."
            if prediction == 1
            else "You are a Healthy Person."
        )

        # Store prediction in history
        prediction_history_db.append({
            'input_data': input_data,
            'result_message': result_message
        })

        # Render the result template with the prediction
        return render_template('result.html', prediction=result_message)
    except Exception as e:
        print('The Exception message is:', e)
        # Return a JSON response with an error message
        return jsonify({'error': f'Something went wrong: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
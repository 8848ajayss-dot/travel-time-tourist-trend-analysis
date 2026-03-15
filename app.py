from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import joblib
import calendar
import logging
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure feedback database and table exist


def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()

# Load dataset
csv_path = "cleaned_travel_dataset_final.csv"
df = pd.read_csv(csv_path)
df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
df.dropna(subset=['Month'], inplace=True)

# Load models
crowd_model = joblib.load("model/crowd_model.pkl")
age_model = joblib.load("model/age_model.pkl")

# Load encoders
le_nationality = joblib.load("model/le_nationality.pkl")
le_gender = joblib.load("model/le_gender.pkl")
le_crowd = joblib.load("model/le_crowd.pkl")
le_age_group = joblib.load("model/le_age_group.pkl")
le_age_destination = joblib.load("model/le_age_destination.pkl")

# Fallback logic for categorical modes


def get_most_frequent(series):
    return series.mode().iloc[0] if not series.mode().empty else None


mapping = df.groupby(['gender', 'Traveler nationality', 'Destination']).agg({
    'purpose': get_most_frequent,
    'Accommodation type': get_most_frequent,
    'Accommodation cost': get_most_frequent,
    'Transportation cost': get_most_frequent
}).reset_index()


@app.route("/")
def index():
    if "user" not in session:
        return redirect("/login")

    stats = {
        'total_tourists': len(df),
        'top_nationalities': df['Traveler nationality'].value_counts().head(3).index.tolist(),
        'top_purposes': df['purpose'].value_counts().head(3).index.tolist(),
        'peak_months': df['Month'].value_counts().head(3).index.tolist(),
        'avg_travel_time': round(df['Duration (days)'].mean(), 1)
    }
    return render_template("index.html", stats=stats)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect("/login")

    prediction = None
    chart_labels = []
    chart_values = []

    if request.method == "POST":
        gender = request.form.get("gender")
        nationality = request.form.get("nationality")
        destination = request.form.get("destination")

        try:
            # Encode inputs
            nat_encoded = le_nationality.transform([nationality])[0]
            gen_encoded = le_gender.transform([gender])[0]
            dest_encoded = le_age_destination.transform([destination])[0]
            features = [[nat_encoded, gen_encoded, dest_encoded]]

            # Predict age group and crowd level
            age_pred = age_model.predict(features)[0]
            crowd_pred = crowd_model.predict(features)[0]

            age_group = le_age_group.inverse_transform([age_pred])[0]
            crowd_level = le_crowd.inverse_transform([crowd_pred])[0]

            # Lookup mapping for purpose, accommodation, and costs
            match = mapping[(mapping['gender'] == gender) &
                            (mapping['Traveler nationality'] == nationality) &
                            (mapping['Destination'] == destination)]

            if not match.empty:
                purpose = match.iloc[0]['purpose']
                accommodation = match.iloc[0]['Accommodation type']
                acc_cost = match.iloc[0]['Accommodation cost']
                trans_cost = match.iloc[0]['Transportation cost']
            else:
                # Fallback if no exact match
                fallback = df[df['Destination'] == destination]
                if fallback.empty:
                    fallback = df[df['Traveler nationality'] == nationality]
                if not fallback.empty:
                    purpose = fallback['purpose'].mode().iloc[0]
                    accommodation = fallback['Accommodation type'].mode(
                    ).iloc[0]
                    acc_cost = fallback['Accommodation cost'].mode().iloc[0]
                    trans_cost = fallback['Transportation cost'].mode().iloc[0]
                else:
                    purpose = accommodation = acc_cost = trans_cost = "Unknown"

            # Parse costs
            try:
                acc_val = float(str(acc_cost).replace("$", "").strip())
            except:
                acc_val = 0
            try:
                trans_val = float(str(trans_cost).replace("$", "").strip())
            except:
                trans_val = 0

            total_cost = acc_val + trans_val
            total_cost = f"{total_cost:.2f}" if total_cost > 0 else "N/A"

            # Map purpose to relevant suggested activities
            purpose_activities_map = {
                "Leisure": "Sightseeing, Local Cuisine, Photography",
                "Business": "Conferences, Meetings, Networking",
                "Adventure": "Hiking, Rafting, Camping",
                "Cultural": "Museum Visits, Cultural Tours, Historical Sites",
                "Shopping": "Mall Visits, Local Markets, Souvenir Hunting",
                "Medical": "Hospital Visits, Wellness Retreats, Resting",
                "Religious": "Temple Visits, Pilgrimages, Meditation",
                "Education": "Campus Tours, Seminars, Study",
                "Transit": "Short Stopovers, Airport Lounges, City Tour",
                "Honeymoon": "Romantic Dinners, Beach Walks, Relaxation"
            }

            preferred_activities = purpose_activities_map.get(
                purpose, "Sightseeing, Local Cuisine, Photography")

            # Prepare chart data
            purpose_counts = df['purpose'].value_counts(normalize=True) * 100
            accommodation_counts = df['Accommodation type'].value_counts(
                normalize=True) * 100
            chart_labels = purpose_counts.index.tolist() + accommodation_counts.index.tolist()
            chart_values = purpose_counts.tolist() + accommodation_counts.tolist()

            monthly_labels = list(calendar.month_name)[1:]
            filtered_df = df[(df['gender'] == gender) &
                             (df['Traveler nationality'] == nationality) &
                             (df['Destination'] == destination)]
            if len(filtered_df) < 5:
                filtered_df = df[(df['Traveler nationality'] == nationality) & (
                    df['Destination'] == destination)]
            if len(filtered_df) < 5:
                filtered_df = df[df['Destination'] == destination]
            if len(filtered_df) < 5:
                filtered_df = df[df['purpose'] == purpose]
            if len(filtered_df) < 5:
                filtered_df = df

            monthly_counts = filtered_df['Month'].value_counts().reindex(
                range(1, 13), fill_value=0).astype(int)
            monthly_values = monthly_counts.tolist()

            prediction = {
                "age_group": age_group,
                "crowd_level": crowd_level,
                "purpose": purpose,
                "accommodation": accommodation,
                "trip_cost": total_cost,
                "activities": preferred_activities,
                "acc_val": acc_val,
                "trans_val": trans_val,
                "gender": gender,
                "nationality": nationality,
                "destination": destination,
                "chart_labels": chart_labels,
                "chart_values": chart_values
            }

            if "history" not in session:
                session["history"] = []
            session["history"].append({
                "gender": gender,
                "nationality": nationality,
                "destination": destination,
                "age_group": age_group,
                "crowd_level": crowd_level,
                "trip_cost": total_cost,
                "purpose": purpose,
                "accommodation": accommodation,
                "activities": preferred_activities,
                "monthly_labels": monthly_labels,
                "monthly_values": monthly_values,
                "chart_labels": chart_labels,
                "chart_values": chart_values
            })
            session.modified = True

        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            prediction = {
                "age_group": "N/A",
                "crowd_level": "N/A",
                "purpose": "N/A",
                "accommodation": "N/A",
                "trip_cost": "N/A",
                "activities": "N/A",
                "acc_val": 0,
                "trans_val": 0,
                "gender": gender,
                "nationality": nationality,
                "destination": destination,
                "error": str(e),
                "chart_labels": [],
                "chart_values": []
            }
            monthly_labels = list(calendar.month_name)[1:]
            monthly_values = [0] * 12
    else:
        monthly_labels = list(calendar.month_name)[1:]
        monthly_values = [0] * 12

    nationalities = sorted(df['Traveler nationality'].dropna().unique())
    genders = sorted(df['gender'].dropna().unique())
    destinations = sorted(df['Destination'].dropna().unique())

    if prediction is None:
        prediction = {}

    if prediction.get("error"):
        return render_template("dashboard.html",
                               prediction=prediction,
                               history=session.get("history", []),
                               nationalities=nationalities,
                               genders=genders,
                               destinations=destinations)

    if request.method == "POST":
        return render_template("result.html",
                               prediction=prediction,
                               monthly_labels=monthly_labels,
                               monthly_values=monthly_values)

    return render_template("dashboard.html",
                           prediction=prediction,
                           history=session.get("history", []),
                           nationalities=nationalities,
                           genders=genders,
                           destinations=destinations)


@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")
    return render_template("history.html", history=session.get("history", []))


@app.route("/clear_history")
def clear_history():
    session.pop("history", None)
    return redirect("/history")


@app.route("/trends")
def trends():
    if "user" not in session:
        return redirect("/login")

    monthly_labels = list(calendar.month_name)[1:]
    monthly_counts = df['Month'].value_counts().reindex(
        range(1, 13), fill_value=0).astype(int)
    monthly_values = monthly_counts.tolist()

    nationality_counts = df['Traveler nationality'].value_counts().head(
        10).to_dict()
    purpose_counts = df['purpose'].value_counts()
    purpose_labels = purpose_counts.index.tolist()
    purpose_values = purpose_counts.tolist()

    monthly_purpose_data = {}
    for purpose in purpose_labels:
        monthly = df[df['purpose'] == purpose]['Month'].value_counts().reindex(
            range(1, 13), fill_value=0).astype(int)
        monthly_purpose_data[purpose] = monthly.tolist()

    return render_template("trends.html",
                           monthly_labels=monthly_labels,
                           monthly_values=monthly_values,
                           nationality_counts=nationality_counts,
                           purpose_labels=purpose_labels,
                           purpose_values=purpose_values,
                           monthly_purpose_labels=monthly_labels,
                           monthly_purpose_data=monthly_purpose_data)


@app.route("/insights")
def insights():
    if "user" not in session:
        return redirect("/login")
    return render_template("insights.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "password123":
            session["user"] = username
            return redirect("/")
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        user_feedback = request.form.get("feedback")
        username = session.get("user", "Anonymous")

        if user_feedback:
            try:
                conn = sqlite3.connect("database.db")
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO feedback (username, comment) VALUES (?, ?)", (username, user_feedback))
                conn.commit()
                conn.close()
                return render_template("feedback.html", message="Thank you for your feedback!")
            except Exception as e:
                logging.error(f"Database error: {e}")
                return render_template("feedback.html", message="Error saving feedback. Please try again.")
        else:
            return render_template("feedback.html", message="Please enter feedback before submitting.")
    return render_template("feedback.html")


if __name__ == "__main__":
    app.run(debug=True)

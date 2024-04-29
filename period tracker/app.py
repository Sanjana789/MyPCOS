from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///periods.db'
db = SQLAlchemy(app)

class Period(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/log_period', methods=['GET', 'POST'])
def log_period():
    if request.method == 'POST':
        start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d')
        new_period = Period(start_date=start_date, end_date=end_date)
        db.session.add(new_period)
        db.session.commit()
        return redirect(url_for('view_history'))
    return render_template('log_period.html')

@app.route('/view_history')
def view_history():
    periods = Period.query.all()
    return render_template('view_history.html', periods=periods)

if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)

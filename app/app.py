from flask import Flask, flash, request, redirect, send_file, render_template
from io import BytesIO
from predict import clear_data, get_predictions
import pandas as pd

ALLOWED_EXTENSIONS = ['csv', 'txt']

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            df = pd.read_csv(file, sep=';')
            clean_df = clear_data(df)
            pred_df = get_predictions(clean_df)
            buffer = BytesIO()
            pred_df.to_csv(buffer, encoding='utf-8')
            buffer.seek(0)
            return send_file(buffer,
                             attachment_filename="pred_df.csv",
                             mimetype='text/csv')
    return render_template('index.html')

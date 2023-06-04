from flask import Flask, render_template, request, url_for
from summarizer import Summarizer

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')

        try:
            summarizer = Summarizer(text)
            summary = summarizer.result()
            return render_template("index.html", summary = summary)
        
        except TypeError():
            print('Invalid text entered')

    else: 
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug = True)
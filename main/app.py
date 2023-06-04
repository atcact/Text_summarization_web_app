# from flask import Flask, render_template, request, jsonify
# from summarizer import Summarizer

# # app = Flask(__name__)

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/summarize', methods=['POST'])
# # def summarize():
# #     # Retrieve the text data from the frontend
# #     text = request.json['text']
# #     num_sentences = request.json['num_sentences']

# #     summarizer = Summarizer(text, num_sentences)
# #     # Return the summary as a JSON response
# #     summary = summarizer.result()
# #     return summary


# # if __name__ == '__main__':
# #     app.run()

# app = Flask(__name__)

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     data = request.get_json()
#     text = data.get('text')
#     num_sentences = data.get('num_sentences')

#     # Perform text summarization logic
#     summarizer = Summarizer(text, num_sentences)
#     summary = summarizer.result()

#     # Return the summary as a JSON response
#     return jsonify({'summary': summary})

# if __name__ == '__main__':
#     app.run(debug=True)

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
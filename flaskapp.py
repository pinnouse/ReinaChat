from flask import Flask
app = Flask(__name__)

@app.route("/<sentence>")
def hello(sentence):
    return "What is going on" + sentence

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
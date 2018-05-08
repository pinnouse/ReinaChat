import bot
import json
from flask import Flask, render_template, request
app = Flask(__name__)

chatbot = bot.Bot()

@app.route("/api")
def api():
    seq, read = chatbot.sentence_to_seq(str(request.args.get('s')))
    out = chatbot.decode_sequence(seq)
    d = {
        'read_sentence': read,
        'out_sentence': out
    }
    return json.dumps(d)

@app.route("/web")
def web():
    sentence, _ = chatbot.sentence_to_seq(str(request.args.get('s')))
    out = ""
    if sentence is not None and sentence is not "":
        out = chatbot.decode_sequence(sentence)
        # print("output:", out)
    return render_template('app.html', output=out)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
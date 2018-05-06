from flask import Flask, render_template, request
import bot
app = Flask(__name__)

@app.route("/api")
def api():
    seq, read = bot.sentence_to_seq(str(request.args.get('s')))
    out = bot.decode_sequence(seq)
    d = {
        'read_sentence': read,
        'out_sentence': out
    }
    return json.dumps(d)

@app.route("/web")
def web():
    print(request.args.get('s'))
    sentence, _ = bot.sentence_to_seq(str(request.args.get('s')))
    out = ""
    if sentence is not None:
        out = bot.decode_sequence(sentence)
    return render_template('app.html', output=out)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

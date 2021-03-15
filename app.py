from flask import Flask, render_template, request, jsonify, make_response
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

    response = make_response('The page named %s doesnt exists.' \
                                %index.html, 404)
    return response


if __name__ == "__main__":

    #app = index()
    app.run(debug=True)
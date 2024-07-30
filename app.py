from flask import Flask, request, jsonify
from services.resource_service import load_data, query_database

app = Flask(__name__)


@app.route('/load', methods=['GET'])
def load():
    load_data()
    return jsonify(success=True)


@app.route('/query', methods=['GET'])
def query():
    question = request.args.get('question')
    results = query_database(question)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)

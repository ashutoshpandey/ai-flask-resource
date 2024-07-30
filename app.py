from flask import Flask, request, jsonify, render_template
from services.resource_service import load_csv, query_service

app = Flask(__name__)


@app.route('/load', methods=['GET'])
def query_database():
    load_csv
    return jsonify(success=True)


@app.route('/query', methods=['POST'])
def query_database():
    query = request.form['query']
    results = query_service(query)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)

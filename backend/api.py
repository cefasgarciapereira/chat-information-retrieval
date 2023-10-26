from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin
from chat import make_a_question

app = Flask(__name__, static_folder='./')
CORS(app, support_credentials=True)

@app.route('/')
def hello_world():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/question', methods=['GET'])
def example_api():
    q = request.args.get('q')
    answer = make_a_question(q)
    data = {'message': answer[0], 'document': answer[1], 'status': 'sucesso'}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

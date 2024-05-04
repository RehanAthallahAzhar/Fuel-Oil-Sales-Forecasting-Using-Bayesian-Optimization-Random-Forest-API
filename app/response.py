from flask import jsonify, make_response


def successPredict(month, values, message):
    res = {
        'month' : month,
        'data' : values,
        'message' : message
    }

    return make_response(jsonify(res)),200

def badRequest(values, message):
    res = {
        'data' : values,
        'message' : message
    }

    return make_response(jsonify(res)),400
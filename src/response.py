from flask import jsonify, make_response


def PredictResp(statuscode, month, values, message):
    res = {
        'month' : month,
        'data' : values,
        'message' : message
    }

    return make_response(jsonify(res)), statuscode

def DataResp(statuscode, values, message):
    res = {
        'data' : values,
        'message' : message
    }

    return make_response(jsonify(res)), statuscode

def JSON(status, message):
    res = {
        'message' : message
    }

    return make_response(jsonify(res)), status
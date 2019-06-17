from flask import Flask, request
from flask_cors import CORS
import json

from textcnn_classify import predict
import logging
app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/', methods=["GET"])
def hello_world():

    data = request.args.get("predict")
    print(data)
    if not data:
        data = ["伊纳宝妙好 亲心妙鲜包猫 RC-43C鲣鱼鲣鱼松鸡肉80G 猫妙鲜包湿粮零食 整盒"]
        content = data
    else:
        content = data.split("==")
    print(content)
    cnn_model = predict.main()
    rlt = cnn_model.predict_test(content)
    print(rlt)
    return json.dumps({"status":"ok", "title":content, "type":rlt})



if __name__ == '__main__':
    app.debug = True
    handler = logging.FileHandler('flask.log')
    app.logger.addHandler(handler)
    app.run(host="0.0.0.0", port="5000")
    # monitor_server()

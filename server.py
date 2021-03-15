import pickle
import ssl
import pandas as pd
import sister
from flask import Flask
from flask import request

app = Flask(__name__)

ssl._create_default_https_context = ssl._create_unverified_context
embedder = sister.MeanEmbedding(lang="en")
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


@app.route("/get_answer", methods=["POST"])
def get_answer():

    content = request.get_json()
    user_replica = pd.DataFrame(embedder(content['user_replica']))
    res = loaded_model.predict(user_replica.T)

    return {"response": str(res[0])}


if __name__ == '__main__':
    app.run()

import os
from flask import Flask, request, jsonify
from flask_restful import Resource, Api,reqparse
from schemas.image import ImageSchema
from marshmallow import ValidationError
from torchvision import transforms
from ml_model_arc import PalmOilLightningClassifier
from PIL import  Image, UnidentifiedImageError
from torch import nn
import torch


app = Flask(__name__)
api = Api(app)


image_schema = ImageSchema()

# class to get all product
class PalmOilDetector(Resource):
    @classmethod
    def post(cls):
        """
        Used to upload an image file.
        It uses JWT to retrieve user information and then saves 
        the image to the user's folder. If there is a filename 
        conflict, it appends a number at the end.
        """

        # data = image_schema.load(request.files)  # {"image":FileStorage}
        try:
            data = image_schema.load(request.files)  # {"image":FileStorage}
        except ValidationError as e:
            return {"message": f"Image Not Found : ({e})"}, 404

        val_trans = transforms.Compose([ transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), 
                                                             (0.5, 0.5, 0.5))
                                          ])
        try:
            img = Image.open(data["image"])
        except UnidentifiedImageError as e:
            return {"message": f"Image Is <None> : ({e})"}, 404

        img = val_trans(img)
        img = img.view((1, *img.shape))
        mapping = {'cuda:0':'cpu'}
        classifier = PalmOilLightningClassifier.load_from_checkpoint("./ml_model/accurate_model.ckpt", map_location=mapping)
        classifier.eval()
        pred_probab = classifier(img)
        print(pred_probab)
        
        #get predictions and loss
        y_hat = int((pred_probab.clone().detach() > 0.2).type(torch.int)[0][0])
        print(y_hat)
        # y_hat = int(pred_probab.argmax(1)[0])
        pred_probab= float(pred_probab[0]) if y_hat == 1 else 1 - float(pred_probab[0])
        print(y_hat, pred_probab)
        return {"prediction" :y_hat, "probability_score":pred_probab}

api.add_resource(PalmOilDetector, "/detect_pailoil")

if __name__ == "__main__":
    app.run(port=7001, debug=True)
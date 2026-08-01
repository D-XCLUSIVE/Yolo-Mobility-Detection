from ultralytics import YOLO

CLASS_NAMES = {
    0: "Crutch",
    1: "Wheel_Mobility_Aid",
    2: "Walker"
}


def load_model():

    model = YOLO("models/best.pt")
    #Override display names
    model.model.names = CLASS_NAMES
    return model


def predict(model, image):

    results = model.predict(
        image,
        conf=0.5,
        verbose=False
    )

    return results

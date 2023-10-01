import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("models/Model Details/v8s-detect-epochs400-imgsz800/weights/best.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("best.pb")

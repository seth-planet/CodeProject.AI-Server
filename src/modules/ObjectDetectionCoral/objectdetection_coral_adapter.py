import sys
import time

# Import the CodeProject.AI SDK. This will add to the PATH var for future imports
sys.path.append("../../SDK/Python")
from common import JSON
from request_data import RequestData
from module_runner import ModuleRunner

# Import the method of the module we're wrapping
from options import Options
from PIL import UnidentifiedImageError, Image

from tpu_runner import TPURunner

opts = Options()
tpu_runner = TPURunner()

def do_detect(options: Options, image: Image, score_threshold: float = 0.5):
    # Run inference
    inference_rs = tpu_runner.process_image(options, image, score_threshold)

    if inference_rs == False:
        return {
            "success"     : False,
            "error"       : "Unable to create interpreter",
            "count"       : 0,
            "predictions" : [],
            "inferenceMs" : 0
        }

    # Get output
    outputs = []
    for obj in inference_rs:
        class_id = obj.id
        caption  = tpu_runner.labels.get(class_id, class_id)
        score    = float(obj.score)
        xmin, ymin, xmax, ymax = obj.bbox

        if score >= score_threshold:
            detection = {
                "confidence": score,
                "label": caption,
                "x_min": xmin,
                "y_min": ymin,
                "x_max": xmax,
                "y_max": ymax,
            }

            outputs.append(detection)

    return {
        "success"     : True,
        "count"       : len(outputs),
        "predictions" : outputs,
        "inferenceMs" : inference_rs[1]
    }

class CoralObjectDetector_adapter(ModuleRunner):

    # async 
    def initialise(self) -> None:
        # if the module was launched outside of the server then the queue name 
        # wasn't set. This is normally fine, but here we want the queue to be
        # the same as the other object detection queues
        if not self.launched_by_server:
            self.queue_name = "objectdetection_queue"

        if self.support_GPU:
            self.support_GPU = self.hasCoralTPU

        if self.support_GPU:
            print("Edge TPU detected")
            self.execution_provider = "TPU"

        device = tpu_runner.init_interpreters(opts)
        if device.upper() == "TPU":
            self.execution_provider = "TPU"
        else:
            self.execution_provider = "CPU"
        
    #async 
    def process(self, data: RequestData) -> JSON:

        # The route to here is /v1/vision/detection

        if data.command == "list-custom":               # list all models available
            return { "success": True, "models": [ 'MobileNet SSD'] }

        if data.command == "detect" or data.command == "custom":
            threshold: float  = float(data.get_value("min_confidence", opts.min_confidence))
            img: Image        = data.get_image(0)

            response = self.do_detection(img, threshold)
        else:
            # await self.report_error_async(None, __file__, f"Unknown command {data.command}")
            self.report_error(None, __file__, f"Unknown command {data.command}")
            response = { "success": False, "error": "unsupported command" }

        return response


    # async 
    def do_detection(self, img: any, score_threshold: float):
        
        start_process_time = time.perf_counter()
    
        try:
        
            result = do_detect(opts, img, score_threshold)

            if not result['success']:
                return {
                    "success"     : False,
                    "predictions" : [],
                    "message"     : '',
                    "error"       : result["error"] if "error" in result else "Unable to perform detection",
                    "count"       : 0,
                    "processMs"   : int((time.perf_counter() - start_process_time) * 1000),
                    "inferenceMs" : result['inferenceMs']
                }
            
            predictions = result["predictions"]
            if len(predictions) > 3:
                message = 'Found ' + (', '.join(det["label"] for det in predictions[0:3])) + "..."
            elif len(predictions) > 0:
                message = 'Found ' + (', '.join(det["label"] for det in predictions))
            elif "error" in result:
                message = result["error"]
            else:
                message = "No objects found"
            
            # print(message)

            return {
                "message"     : message,
                "count"       : result["count"],
                "predictions" : result['predictions'],
                "success"     : result['success'],
                "processMs"   : int((time.perf_counter() - start_process_time) * 1000),
                "inferenceMs" : result['inferenceMs']
            }

        except UnidentifiedImageError as img_ex:
            # await self.report_error_async(img_ex, __file__, "The image provided was of an unknown type")
            self.report_error(img_ex, __file__, "The image provided was of an unknown type")
            return { "success": False, "error": "invalid image file" }

        except Exception as ex:
            # await self.report_error_async(ex, __file__)
            self.report_error(ex, __file__)
            return { "success": False, "error": "Error occurred on the server"}


if __name__ == "__main__":
    CoralObjectDetector_adapter().start_loop()

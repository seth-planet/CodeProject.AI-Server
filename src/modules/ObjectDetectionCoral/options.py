import os
import logging

try:
    from module_options import ModuleOptions
except ImportError:
    logging.warning("Unable to import ModuleOptions, running with defaults")
    ModuleOptions = None

class Settings:
    def __init__(self, resolution, std_model_name, tpu_model_name, labels_name, tpu_segment_names):
        self.resolution     = resolution
        self.cpu_model_name = std_model_name
        self.tpu_model_name = tpu_model_name
        self.labels_name    = labels_name
        self.tpu_segment_names = tpu_segment_names

class Options:

    def __init__(self):
    
        # -------------------------------------------------------------------------
        # Setup constants

        # Models at:
        # https://coral.ai/models/object-detection/
        # https://github.com/MikeLud/CodeProject.AI-Custom-IPcam-Models/
        self.MODEL_SETTINGS = {
            "YOLOv5": {
                "large":  Settings(448, 'yolov5l-int8.tflite',
                                        'yolov5l-int8_edgetpu.tflite',
                                        'coco_labels.txt',
                                        [['yolov5l-int8_edgetpu_segment_0_of_3_edgetpu.tflite',
                                          'yolov5l-int8_edgetpu_segment_1_of_3_edgetpu.tflite',
                                          'yolov5l-int8_edgetpu_segment_2_of_3_edgetpu.tflite'],
                                         ['yolov5l-int8_edgetpu_segment_0_of_2_edgetpu.tflite',
                                          'yolov5l-int8_edgetpu_segment_1_of_2_edgetpu.tflite']]),
                "medium": Settings(448, 'yolov5m-int8.tflite',
                                        'yolov5m-int8_edgetpu.tflite',
                                        'coco_labels.txt',
                                        [['yolov5m-int8_edgetpu_segment_0_of_2_edgetpu.tflite',
                                          'yolov5m-int8_edgetpu_segment_1_of_2_edgetpu.tflite']]),
                "small": Settings(448,  'yolov5s-int8.tflite',
                                        'yolov5s-int8_edgetpu.tflite',
                                        'coco_labels.txt', []),
                "tiny": Settings(448,   'yolov5n-int8.tflite',
                                        'yolov5n-int8_edgetpu.tflite',
                                        'coco_labels.txt', [])},
            "EfficientDet-Lite": {
                # Large: EfficientDet-Lite3x 90 objects COCO	640x640x3 	2 	197.0 ms 	43.9% mAP
                "large":  Settings(640, 'efficientdet_lite3x_640_ptq.tflite',
                                        'efficientdet_lite3x_640_ptq_edgetpu.tflite',
                                        'coco_labels.txt',
                                        [['efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite',
                                          'efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite']]),
                # Medium: EfficientDet-Lite3 90 objects	512x512x3 	2 	107.6 ms 	39.4% mAP
                "medium": Settings(512, 'efficientdet_lite3_512_ptq.tflite',
                                        'efficientdet_lite3_512_ptq_edgetpu.tflite',
                                        'coco_labels.txt', []),
                # Small: EfficientDet-Lite2 90 objects COCO	448x448x3 	2 	104.6 ms 	36.0% mAP
                "small": Settings(448,  'efficientdet_lite2_448_ptq.tflite',
                                        'efficientdet_lite2_448_ptq_edgetpu.tflite',
                                        'coco_labels.txt', []),

                # Tiny: EfficientDet-Lite1 90 objects COCO	384x384x3 	2 	56.3 ms 	34.3% mAP
                "tiny": Settings(384,   'efficientdet_lite1_384_ptq.tflite',
                                        'efficientdet_lite1_384_ptq_edgetpu.tflite',
                                        'coco_labels.txt', [])},
            "MobileNet SSD": {
                # Large: SSD/FPN MobileNet V1 90 objects, COCO 640x640x3    TF-lite v2    229.4 ms    31.1% mAP
                "large":  Settings(640, 'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq.tflite',
                                        'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_edgetpu.tflite',
                                        'coco_labels.txt',
                                        [['tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite',
                                          'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite']]),
                # Medium: SSDLite MobileDet   90 objects, COCO 320x320x3    TF-lite v1    9.1 ms 	32.9% mAP
                "medium": Settings(320, 'ssdlite_mobiledet_coco_qat_postprocess.tflite',
                                        'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',
                                        'coco_labels.txt', []),
                # Small: SSD MobileNet V2 90 objects, COCO 300x300x3    TF-lite v2    7.6 ms    22.4% mAP
                "small": Settings(300,  'tf2_ssd_mobilenet_v2_coco17_ptq.tflite',
                                        'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite',
                                        'coco_labels.txt', []),

                # Tiny: MobileNet V2 90 objects, COCO 300x300x3    TF-lite v2 Quant
                "tiny": Settings(300,   'ssd_mobilenet_v2_coco_quant_postprocess.tflite',
                                        'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                                        'coco_labels.txt', [])}
                                        }

        self.MIN_CONFIDENCE = 0.5
        
        # -------------------------------------------------------------------------
        # Setup values

        self._show_env_variables = True

        self.module_path = "."
        self.models_dir = "."
        if ModuleOptions:
            self.module_path    = ModuleOptions.module_path
            self.models_dir     = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR", f"{self.module_path}/assets"))

            self.min_confidence = float(ModuleOptions.getEnvVariable("MIN_CONFIDENCE", self.MIN_CONFIDENCE))

        self.sleep_time     = 0.01
        
        # smaller number results in more tiles generated
        self.downsample_by  = 5.2
        self.tile_overlap   = 15
        self.iou_threshold  = 0.1

        self.set_model('MobileNet SSD')

            
    def set_model(self, model_name):
        self.model_name = model_name
        self.model_size = "Small"
        if ModuleOptions:
            self.model_size     = ModuleOptions.getEnvVariable("MODEL_SIZE", "Small")   # small, medium, large

        # Normalise input
        self.model_size     = self.model_size.lower()
        if self.model_size not in [ "tiny", "small", "medium", "large" ]:
            self.model_size = "small"

        # Get settings
        settings = self.MODEL_SETTINGS[model_name][self.model_size]   
        self.cpu_model_name = settings.cpu_model_name
        self.tpu_model_name = settings.tpu_model_name
        self.labels_name    = settings.labels_name
        self.tpu_segment_names = settings.tpu_segment_names

        # pre-chew
        self.model_cpu_file = os.path.normpath(os.path.join(self.models_dir, self.cpu_model_name))
        self.model_tpu_file = os.path.normpath(os.path.join(self.models_dir, self.tpu_model_name))
        self.label_file     = os.path.normpath(os.path.join(self.models_dir, self.labels_name))
        
        self.tpu_segment_files = [[os.path.normpath(os.path.join(self.models_dir, n)) for n in name_list] for name_list in self.tpu_segment_names]

        # -------------------------------------------------------------------------
        # dump the important variables
        if self._show_env_variables:
            logging.info(f"Debug: MODULE_PATH:    {self.module_path}")
            logging.info(f"Debug: MODELS_DIR:     {self.models_dir}")
            logging.info(f"Debug: MODEL_SIZE:     {self.model_size}")
            logging.info(f"Debug: CPU_MODEL_NAME: {self.cpu_model_name}")
            logging.info(f"Debug: TPU_MODEL_NAME: {self.tpu_model_name}")



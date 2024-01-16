import os
try:
    from module_options import ModuleOptions
except ImportError:
    print("Unable to import ModuleOptions, running with defaults")
    class ModuleOptions:
        module_path = '.'
        def getEnvVariable(a, b):
            return b

class Settings:
    def __init__(self, resolution, std_model_name, tpu_model_name, labels_name, tpu_segment_names):
        self.resolution        = resolution
        self.cpu_model_name    = std_model_name
        self.tpu_model_name    = tpu_model_name
        self.labels_name       = labels_name
        self.tpu_segment_names = tpu_segment_names

class Options:

    def __init__(self):

        # ----------------------------------------------------------------------
        # Setup constants

        # Models at:
        # https://coral.ai/models/object-detection/
        # https://github.com/MikeLud/CodeProject.AI-Custom-IPcam-Models/
        self.MODEL_SETTINGS = {
            "YOLOv5": {
                "large":  Settings(448, 'yolov5l-int8.tflite',
                                        'yolov5l-int8_edgetpu.tflite',
                                        'coco_labels.txt',
                                        [['yolov5l-int8_edgetpu_segment_0_of_5_edgetpu.tflite',
                                          'yolov5l-int8_edgetpu_segment_1_of_5_edgetpu.tflite',
                                          'yolov5l-int8_edgetpu_segment_2_of_5_edgetpu.tflite',
                                          'yolov5l-int8_edgetpu_segment_3_of_5_edgetpu.tflite',
                                          'yolov5l-int8_edgetpu_segment_4_of_5_edgetpu.tflite'],
                                         ['yolov5l-int8_edgetpu_segment_0_of_3_edgetpu.tflite',
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
                                        'coco_labels.txt',
                                        [['yolov5s-int8_edgetpu_segment_0_of_2_edgetpu.tflite',
                                          'yolov5s-int8_edgetpu_segment_1_of_2_edgetpu.tflite']]),
                "tiny": Settings(448,   'yolov5n-int8.tflite',
                                        'yolov5n-int8_edgetpu.tflite',
                                        'coco_labels.txt', [])},
            "EfficientDet-Lite": {
                # Large: EfficientDet-Lite3x 90 objects COCO    640x640x3     2     197.0 ms     43.9% mAP
                "large":  Settings(640, 'efficientdet_lite3x_640_ptq.tflite',
                                        'efficientdet_lite3x_640_ptq_edgetpu.tflite',
                                        'coco_labels.txt',
                                        [['efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite',
                                          'efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite']]),
                # Medium: EfficientDet-Lite3 90 objects    512x512x3     2     107.6 ms     39.4% mAP
                "medium": Settings(512, 'efficientdet_lite3_512_ptq.tflite',
                                        'efficientdet_lite3_512_ptq_edgetpu.tflite',
                                        'coco_labels.txt', []),
                # Small: EfficientDet-Lite2 90 objects COCO    448x448x3     2     104.6 ms     36.0% mAP
                "small": Settings(448,  'efficientdet_lite2_448_ptq.tflite',
                                        'efficientdet_lite2_448_ptq_edgetpu.tflite',
                                        'coco_labels.txt',
                                        [['efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite',
                                          'efficientdet_lite2_448_ptq_segment_1_of_3_edgetpu.tflite',
                                          'efficientdet_lite2_448_ptq_segment_2_of_3_edgetpu.tflite']]),
                # Tiny: EfficientDet-Lite1 90 objects COCO    384x384x3     2     56.3 ms     34.3% mAP
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
                # Medium: SSDLite MobileDet   90 objects, COCO 320x320x3    TF-lite v1    9.1 ms     32.9% mAP
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
                                        
        self.ENABLE_MULTI_TPU                   = True
        
        self.MIN_CONFIDENCE                     = 0.5
        self.INTERPRETER_LIFESPAN_SECONDS       = 3600.0
        self.WATCHDOG_IDLE_SECS                 = 5.0       # To be added to non-multi code
        self.MAX_IDLE_SECS_BEFORE_RECYCLE       = 60.0      # To be added to non-multi code
        self.WARN_TEMPERATURE_THRESHOLD_CELSIUS = 80        # PCIe only

        self.MAX_PIPELINE_QUEUE_LEN             = 1000      # Multi-only
        self.TILE_OVERLAP                       = 15        # Multi-only.
        self.DOWNSAMPLE_BY                      = 5.2       # Multi-only. Smaller number results in more tiles generated
        self.IOU_THRESHOLD                      = 0.1       # Multi-only

        # ----------------------------------------------------------------------
        # Setup values

        self._show_env_variables = True

        self.module_path    = ModuleOptions.module_path
        self.models_dir     = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR", f"{self.module_path}/assets"))

        # custom_models_dir = os.path.normpath(ModuleOptions.getEnvVariable("CUSTOM_MODELS_DIR", f"{module_path}/custom-models"))

        self.use_multi_tpu  = ModuleOptions.getEnvVariable("CPAI_CORAL_MULTI_TPU", str(self.ENABLE_MULTI_TPU)).lower() == "true"
        self.min_confidence = float(ModuleOptions.getEnvVariable("MIN_CONFIDENCE", self.MIN_CONFIDENCE))

        self.sleep_time     = 0.01

        # For multi-TPU tiling. Smaller number results in more tiles generated
        self.downsample_by  = float(ModuleOptions.getEnvVariable("CPAI_CORAL_DOWNSAMPLE_BY", self.DOWNSAMPLE_BY))
        self.tile_overlap   = int(ModuleOptions.getEnvVariable("CPAI_CORAL_TILE_OVERLAP",    self.TILE_OVERLAP))
        self.iou_threshold  = float(ModuleOptions.getEnvVariable("CPAI_CORAL_IOU_THRESHOLD", self.IOU_THRESHOLD))

        # Maybe - perhaps! - we need shorter var names
        self.watchdog_idle_secs           = float(ModuleOptions.getEnvVariable("CPAI_CORAL_WATCHDOG_IDLE_SECS",           self.WATCHDOG_IDLE_SECS))
        self.interpreter_lifespan_secs    = float(ModuleOptions.getEnvVariable("CPAI_CORAL_INTERPRETER_LIFESPAN_SECONDS", self.INTERPRETER_LIFESPAN_SECONDS))
        self.max_idle_secs_before_recycle = float(ModuleOptions.getEnvVariable("CPAI_CORAL_MAX_IDLE_SECS_BEFORE_RECYCLE", self.MAX_IDLE_SECS_BEFORE_RECYCLE))
        self.max_pipeline_queue_length    = int(ModuleOptions.getEnvVariable("CPAI_CORAL_MAX_PIPELINE_QUEUE_LEN",         self.MAX_PIPELINE_QUEUE_LEN))
        self.warn_temperature_thresh_C    = int(ModuleOptions.getEnvVariable("CPAI_CORAL_WARN_TEMPERATURE_THRESHOLD_CELSIUS", self.WARN_TEMPERATURE_THRESHOLD_CELSIUS))

        self.set_model('MobileNet SSD')

            
    def set_model(self, model_name):
        self.model_name = model_name
        self.model_size = "Small"
        if ModuleOptions:
            self.model_size     = ModuleOptions.getEnvVariable("MODEL_SIZE", "Small")   # small, medium, large

        # Normalise input
        self.model_size     = self.model_size.lower()

        """
        With models MobileNet SSD, EfficientDet-Lite, and YOLOv5, we have three
        classes of model. The first is basically designed to work in concert
        with the Edge TPU and are compatible with the Dev Board Micro. They are
        very fast and don't require additional CPU resources. The YOLOv5 models
        should be directly comparable with other CPAI modules running YOLOv5.
        They should be high-quality, but are not designed with the Edge TPU in
        mind and rely more heavily on the CPU. The EfficientDet-Lite models are
        in between: not as modern as YOLOv5, but less reliant on the CPU.
        
        Each class of model is broken into four sizes depending on the
        intensity of the workload.
        """
        model_valid = self.model_size in [ "tiny", "small", "medium", "large" ]
        if not model_valid:
            self.model_size = "small"

        self.use_YOLO = ModuleOptions.getEnvVariable("CPAI_CORAL_USE_YOLO", "false").lower() == "true"
        # if not self.use_multi_tpu: # Doesn't seem to work on non-multi-TPU code
        #     self.use_YOLO = False
        self.use_YOLO = False        # Doesn't seem to work at all for me :(

        # Get settings
        settings = self.MODEL_SETTINGS[model_name][self.model_size]
        self.cpu_model_name = settings.cpu_model_name
        self.tpu_model_name = settings.tpu_model_name
        self.labels_name    = settings.labels_name
        self.tpu_segment_names = settings.tpu_segment_names

        # pre-chew
        self.model_cpu_file    = os.path.normpath(os.path.join(self.models_dir, self.cpu_model_name))
        self.model_tpu_file    = os.path.normpath(os.path.join(self.models_dir, self.tpu_model_name))
        self.label_file        = os.path.normpath(os.path.join(self.models_dir, self.labels_name))
        self.tpu_segment_files = [os.path.normpath(os.path.join(self.models_dir, n)) for n in self.tpu_segment_names]

        # ----------------------------------------------------------------------
        # dump the important variables

        if self._show_env_variables:
            print(f"Debug: MODULE_PATH:    {self.module_path}")
            print(f"Debug: MODELS_DIR:     {self.models_dir}")
            print(f"Debug: MODEL_SIZE:     {self.model_size}")
            print(f"Debug: CPU_MODEL_NAME: {self.cpu_model_name}")
            print(f"Debug: TPU_MODEL_NAME: {self.tpu_model_name}")
            

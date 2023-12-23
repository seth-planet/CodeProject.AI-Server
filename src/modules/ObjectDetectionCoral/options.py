import os
from module_options import ModuleOptions

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

        # Models at https://coral.ai/models/object-detection/
        self.MODEL_SETTINGS = {
            # Large: EfficientDet-Lite3x    90 objects, COCO 640x640x3    TF-lite v2    197.0 ms    43.9% mAP
            "large":  Settings(640, 'efficientdet_lite3x_640_ptq.tflite',
                                    'efficientdet_lite3x_640_ptq_edgetpu.tflite',
                                    'coco_labels.txt',
                                    ['efficientdet_lite3x_640_ptq_segment_0_of_3_edgetpu.tflite',
                                     'efficientdet_lite3x_640_ptq_segment_1_of_3_edgetpu.tflite',
                                     'efficientdet_lite3x_640_ptq_segment_2_of_3_edgetpu.tflite']),
                                    
            # Medium: EfficientDet-Lite2    90 objects, COCO 448x448x3    TF-lite v2    104.6 ms    36.0%  mAP
            # Note: The compiler had trouble with EfficientDet-Lite3 and a large chunk didn't fit on
            # the TPU anyway, so we're using Lite2 since it fits well on 2 TPUs.
            "medium": Settings(448, 'efficientdet_lite2_448_ptq.tflite',
                                    'efficientdet_lite2_448_ptq_edgetpu.tflite',
                                    'coco_labels.txt',
                                    ['efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite',
                                     'efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite']),

            # Small: EfficientDet-Lite1     90 objects, COCO 384x384x3    TF-lite v2    56.3 ms     34.3% mAP
            "small": Settings(384,  'efficientdet_lite1_384_ptq.tflite',
                                    'efficientdet_lite1_384_ptq_edgetpu.tflite',
                                    'coco_labels.txt',
                                    None),
            '''
            # Small: SSD MobileDet      90 objects, COCO 320x320x3    TF-lite v2    9.1 ms      32.9% mAP
            # This seems redundant with 'small', but faster
            "small": Settings(320,  'ssdlite_mobiledet_coco_qat_postprocess.tflite',
                                    'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',
                                    'coco_labels.txt',
                                    None),'''
            '''
            # Small: SSD MobileNet V2   90 objects, COCO 300x300x3    TF-lite v2    7.6 ms      22.4% mAP
            # This seems redundant with 'tiny' but lower precision
            "small": Settings(300,  'tf2_ssd_mobilenet_v2_coco17_ptq.tflite',
                                    'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite',
                                    'coco_labels.txt',
                                    None),'''

            # Tiny: MobileNet V2            90 objects, COCO 300x300x3    TF-lite v2    7.3 ms      25.6% mAP
            "tiny": Settings(300,   'ssd_mobilenet_v2_coco_quant_postprocess.tflite',
                                    'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                                    'coco_labels.txt',
                                    None),
        }

        self.NUM_THREADS    = 1
        self.MIN_CONFIDENCE = 0.5
        
        # -------------------------------------------------------------------------
        # Setup values

        self._show_env_variables = True

        self.module_path    = ModuleOptions.module_path
        self.models_dir     = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR", f"{self.module_path}/assets"))
        self.model_size     = ModuleOptions.getEnvVariable("MODEL_SIZE", "Small")   # small, medium, large

        # custom_models_dir = os.path.normpath(ModuleOptions.getEnvVariable("CUSTOM_MODELS_DIR", f"{module_path}/custom-models"))

        self.num_threads    = int(ModuleOptions.getEnvVariable("NUM_THREADS",      self.NUM_THREADS))
        self.min_confidence = float(ModuleOptions.getEnvVariable("MIN_CONFIDENCE", self.MIN_CONFIDENCE))

        self.sleep_time     = 0.01
        
        # smaller number results in more tiles generated
        self.downsample_by  = 5.8
        self.tile_overlap   = 15
        self.iou_threshold  = 0.1

        # Start with fewer processes, but we can scale up as needed
        # Processes are pretty heavy on memory.
        self.resize_processes = 1

        # Normalise input
        self.model_size     = self.model_size.lower()
        if self.model_size not in [ "tiny", "small", "medium", "large" ]:
            self.model_size = "small"

        # Get settings
        settings = self.MODEL_SETTINGS[self.model_size]   
        self.cpu_model_name = settings.cpu_model_name
        self.tpu_model_name = settings.tpu_model_name
        self.labels_name    = settings.labels_name
        if any(settings.tpu_segment_names):
            self.tpu_segment_names = settings.tpu_segment_names
        else:
            self.tpu_segment_names = []

        # pre-chew
        self.model_cpu_file = os.path.normpath(os.path.join(self.models_dir, self.cpu_model_name))
        self.model_tpu_file = os.path.normpath(os.path.join(self.models_dir, self.tpu_model_name))
        self.label_file     = os.path.normpath(os.path.join(self.models_dir, self.labels_name))
        self.tpu_segment_files = [os.path.normpath(os.path.join(self.models_dir, n)) for n in self.tpu_segment_names]

        # -------------------------------------------------------------------------
        # dump the important variables

        if self._show_env_variables:
            print(f"Debug: MODULE_PATH:    {self.module_path}")
            print(f"Debug: MODELS_DIR:     {self.models_dir}")
            print(f"Debug: MODEL_SIZE:     {self.model_size}")
            print(f"Debug: CPU_MODEL_NAME: {self.cpu_model_name}")
            print(f"Debug: TPU_MODEL_NAME: {self.tpu_model_name}")

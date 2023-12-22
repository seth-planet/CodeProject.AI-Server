# Lint as: python3
# Copyright 2023 Seth Price seth.pricepages@gmail.com
# Parts copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import os
import time
import logging
import copy

from datetime import datetime

import numpy as np

# For Linux we have installed the pycoral libs via apt-get, not PIP in the venv,
# So make sure the interpreters can find the coral libraries
#if platform.system() == "Linux":
#    sys.path.insert(0, "/usr/lib/python3.9/site-packages/")

from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import list_edge_tpus
import pycoral.pipeline.pipelined_model_runner as pipeline

from options import Options


INTERPRETER_LIFESPAN_SECONDS = 3600  # Refresh the interpreters once an hour

# Don't let these fill indefinitely until something more unexpected goes wrong.
# 1000 is arbitrarily chosen to block before things get ugly.
MAX_PIPELINE_QUEUE_LEN = 1000

# Warn if any TPU reads above this temperature C
# https://coral.ai/docs/pcie-parameters/#use-dynamic-frequency-scaling
WARN_TEMPERATURE_THRESHOLD = 80

# Nothing should ever sit in a queue longer than this many seconds.
MAX_WAIT_TIME = 60.0


class TPURunners(object):

    def __init__(self):
        # Refresh the interpreters once an hour
        interpreter_lifespan_secs = 3600

        interpreters        = None  # The model interpreters
        interpreter_created = None  # When were the interpreters created?
        labels              = None  # set of labels for this model
        runners             = None  # Pipeline(s) to run the model

        next_runner_idx     = 0     # Which runner to execute on next?
        postboxes           = None # Store output
        postmen             = None
        runner_lock         = threading.Lock()
        

        # Find the temperature file
        # https://coral.ai/docs/pcie-parameters/
        temp_fname_formats = ['/dev/apex_{}/temp',
                              '/sys/class/apex/apex_{}/temp']
        self.temp_fname_format = None
        tpu_count = len(list_edge_tpus())

        for fn in temp_fname_formats:
            for i in tpu_count:
                if os.path.exists(fn.format(i)):
                    self.temp_fname_format = fn
                    logging.debug(f"Found temperature file at :"+fn.format(i))
                    return
        logging.debug("Unable to find temperature file")
                    
                    
    def _post_service(self, r: pipeline.PipelinedModelRunner, q: queue.Queue):
        while True:
            # Get the next result from runner N
            rs = r.pop()
            
            # Exit if the pipeline is done
            if not rs:
                return
                
            # Get the next receiving queue and deliver the results.
            # Neither of these get() or put() operations should be blocking
            # in the normal case.
            # We may need to use copy.copy(rs) here if we see:
            # RuntimeError: There is at least 1 reference to internal data...
            # But I think the use of runners will be enough to fix the problem.
            q.get(timeout=MAX_WAIT_TIME).put(rs,
                                             timeout=MAX_WAIT_TIME)


    # Must be called while holding runner_lock
    def _init_interpreters(self, options: Options) -> str:

        self.interpreters = []
        self.runners = []
        self.segment_count = max(1, len(options.tpu_segment_files))
        
        # Only allocate the TPUs we will use
        tpu_count = len(list_edge_tpus()) # segment_count
        tpu_count *= segment_count
       
        # Read labels
        self.labels = None
        self.labels = read_label_file(options.label_file) \
                                                if options.label_file else {}

        # Initialize TF-Lite interpreters.
        device = ""
        try:
            device = "tpu"

            for i in range(tpu_count):
                # Alloc all segments into all TPUs, but no more than that.
                if segment_count > 1:
                    tpu_segment_file = \
                        options.tpu_segment_files[i % segment_count]
                else:
                    tpu_segment_file = options.model_tpu_file
                
                interpreters.append(make_interpreter(
                                            tpu_segment_file,
                                            device=":{}".format(i),
                                            delegate=None))
            logging.debug("Loaded {} TPUs".format(tpu_count))

            # Fallback to CPU
            if not any(interpreters):
                segment_count = 1
                device = "cpu"
                self.interpreters = [make_interpreter(
                                        options.model_cpu_file,
                                        device="cpu",
                                        delegate=None)]

        except Exception as ex:
            try:
                logging.exception(
                    "CAUGHT EXCEPTION: Unable to find or initialize the "
                    "Coral TPU. Falling back to CPU-only.")
                
                # Fallback even more
                segment_count = 1
                device = "cpu"
                self.interpreters = [make_interpreter(
                                        options.model_cpu_file,
                                        device="cpu",
                                        delegate=None)]
            except Exception as ex:
                logging.warning("Error creating interpreter: " + str(ex))
                self.interpreters = None
        
        # Womp womp
        if not any(self.interpreters):
            return ""

        # Initialize interpreters
        for i in self.interpreters:
            self.interpreters.allocate_tensors()

        self.interpreter_created = datetime.now()
        
        # Initialize runners/pipelines
        for i in range(0, tpu_count, segment_count):
            self.runners.append(
                pipeline.PipelinedModelRunner(
                    self.interpreters[i:i+segment_count]))
            
            self.runners.set_input_queue_size(MAX_PIPELINE_QUEUE_LEN)
            self.runners.set_output_queue_size(MAX_PIPELINE_QUEUE_LEN)

        # Setup postal queue
        self.postboxs = []
        self.postmen = []

        for r in for self.runners:
            # Start the queue. Set a size limit to keep the queue from going
            # off the rails. An OOM condition will bring everything else down.
            q = queue.Queue(maxsize=MAX_PIPELINE_QUEUE_LEN)
            self.postboxes.append(q)

            # Start the receiving worker
            t = threading.Thread(target=self._post_service, args=[r, q])
            t.start()
            self.postmen.append(t)

        # Get input and output tensors.
        input_details  = self.get_input_details()
        output_details = self.get_output_details()

        # Print debug
        logging.debug(f"TPU & segment counts: {} & {}\n".format(tpu_count, segment_count)
        logging.debug(f"Interpreter count: {}\n".format(len(self.interpreters))
        logging.debug(f"Input details: {input_details}\n")
        logging.debug(f"Output details: {output_details}\n")

        return device


    def _periodic_check(self, options: Options):
        # Check temperatures
        msg = "Core {} is {} Celsius and will likely be throttled"
        if self.temp_fname_format != None:
            for i in len(self.interpreters):
                if os.path.exists(fn.format(i)):
                    with open(fn.format(i), "r") as fp:
                        # Convert from milidegree C to degree C
                        temp = int(fp.read()) // 1000
                        
                        if WARN_TEMPERATURE_THRESHOLD <= temp:
                            logging.warning(msg.format(i, temp))
    
        # Once an hour, refresh the interpreters
        if any(self.interpreters):
            seconds_since_created = \
                (datetime.now() - self.interpreter_created).total_seconds()
                
            if seconds_since_created > interpreter_lifespan_secs:
                logging.info("Refreshing the Tensorflow Interpreters")

                # Close all existing work before destroying...
                with self.runner_lock:
                    self._delete()
                    
                    # Re-init while we still have the lock
                    self._init_interpreters(options)

        # (Re)start them if needed
        if not any(self.interpreters):
            with self.runner_lock:
                self._init_interpreters(options)

        return any(self.interpreters)


    def _delete(self):
        # Close each of the pipelines
        for i in range(len(self.runners)):
            self.runners[i].push({})

        # The above should trigger all workers to end. If we are deleting in an
        # off-the-rails context, it's likely that something will have trouble
        # closing out its work.
        for t in self.postmen:
            t.join(timeout=MAX_WAIT_TIME)
            if t.is_alive():
                logging.warning("Thread didn't join!")
        
        # Delete
        self.postmen        = None
        self.postboxes      = None
        self.runners        = None
        self.interpreters   = None


    def process_image(self,
                      options:Options,
                      image: Image,
                      score_threshold: float):

        if not self._periodic_check(options):
            return False

        all_objects = []
        all_queues = []
        name = self.get_input_details()['name']
        
        # Potentially resize & pipeline a number of tiles
        for rs_image, rs_loc in self._get_tiles(options, image):
            rs_queue = queue.Queue()
            all_queues.append((rs_queue, rs_loc))

            with self.runner_lock:
                # Push the resampled image and where to put the results.
                # There shouldn't be any wait time to put() into the queue, so
                # if we are blocked here, something has gone off the rails.
                self.runners[self.next_runner_idx].push({name: rs_image})
                self.postboxes[self.next_runner_idx].put(rs_queue,
                                                         timeout=MAX_WAIT_TIME)

                # Increment the next runner to use
                self.next_runner_idx = \
                                (1 + self.next_runner_idx) % len(self.runners)

        # Fetch details needed for pipeline output
        score_scale, zero_point = self.get_output_details()['quantization']

        # Wait for the results here
        start_inference_time = time.perf_counter()
        for rs_queue, rs_loc in all_queues:
            # Wait for results
            # We may have to wait a few seconds at most, but I'd expect the
            # pipeline to clear fairly quickly.
            result = rs_queue.get(timeout=MAX_WAIT_TIME)
            assert result
            
            score_values, boxes, count, class_ids = result.values()
            scores = score_scale * (score_values[0].astype(np.int64) - zero_point)
            
            sx, sy = rs_loc[2], rs_loc[3]

            # Create Objects for each valid result
            for i in range(int(count[0])):
                if scores[i] < score_threshold:
                    continue
                    
                ymin, xmin, ymax, xmax = boxes[0][i]
                
                bbox = detect.BBox(xmin=xmin,
                                   ymin=ymin,
                                   xmax=xmax,
                                   ymax=ymax).scale(sx, sy)
                                      
                bbox.translate(rs_loc[0], rs_loc[1])
                                          
                all_objects.append(detect.Object(id=int(class_ids[0][i]),
                                                 score=float(scores[i]),
                                                 bbox=bbox.map(int))

        end_time = int((time.perf_counter() - start_inference_time) * 1000)

        # Remove duplicate objects
        idxs = self._non_max_suppression(all_objects, options.iou_threshold)
        
        return ([all_objects[i] for i in idxs], end_time)
        
        
    def _non_max_suppression(self, objects, threshold):
        """Returns a list of indexes of objects passing the NMS.

        Args:
        objects: result candidates.
        threshold: the threshold of overlapping IoU to merge the boxes.

        Returns:
        A list of indexes containings the objects that pass the NMS.
        """
        if len(objects) == 1:
            return [0]

        boxes = np.array([o.bbox for o in objects])
        xmins = boxes[:, 0]
        ymins = boxes[:, 1]
        xmaxs = boxes[:, 2]
        ymaxs = boxes[:, 3]

        areas = (xmaxs - xmins) * (ymaxs - ymins)
        scores = [o.score for o in objects]
        idxs = np.argsort(scores)

        selected_idxs = []
        while idxs.size != 0:

            selected_idx = idxs[-1]
            selected_idxs.append(selected_idx)

            overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
            overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
            overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
            overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

            w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
            h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

            intersections = w * h
            unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
            ious = intersections / unions

            idxs = np.delete(
                idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

        return selected_idxs


    def _get_tiles(self, options:Options, image: Image):
        _, m_height, m_width, _ = self.get_input_details()['shape']
        i_width, i_height = image.size

        # What tile dim do we want?
        tiles_x = max(1, round(i_width / (options.downsample_by * m_width)))
        tiles_y = max(1, round(i_height / (options.downsample_by * m_height)))
        logging.debug("Chunking to {} x {} tiles".format(tiles_x, tiles_y))

        # Resample image to this size
        resamp_x = m_width  + (tiles_x - 1) * (m_width  - options.tile_overlap)
        resamp_y = m_height + (tiles_y - 1) * (m_height - options.tile_overlap)
        logging.debug("Resizing to {} x {} for tiling".format(resamp_x, resamp_y))

        # Chop & resize image piece
        resamp_img = image.convert('RGB').resize((resamp_x, resamp_y),
                                                 Image.LANCZOS)

        # Normalize pixel values
        params = self.get_input_details()['quantization_parameters']
        scale = params['scales']
        zero_point = params['zero_points']

        # Do chunking
        for x_off in range(0, resamp_x, m_width - options.tile_overlap):
            for y_off in range(0, resamp_y, m_height - options.tile_overlap):
                cropped_arr = crop((x_off,
                                    y_off,
                                    x_off + m_width,
                                    y_off + m_height)).asarray()
                logging.debug("Resampled image tile {}".format(cropped_arr.size))
            
                # Normalize input image
                normalized_input = zero_point + \
                    (cropped_arr.astype('float32') - cropped_arr.mean()) /
                                                (cropped_arr.std() * scale * 2)
                np.clip(normalized_input, 0, 255, out=normalized_input)

                # Print image stats
                logging.debug('Input Min: %.3f, Max: %.3f' %
                                    (cropped_arr.min(), cropped_arr.max()))
                logging.debug('Normalized Min: %.3f, Max: %.3f' %
                            (normalized_input.min(), normalized_input.max()))
                logging.debug('Normalized Mean: %.3f, Standard Deviation: %.3f' %
                            (normalized_input.mean(), normalized_input.std()))

                yield (normalized_input.astype(np.uint8),
                       (x_off, y_off, i_width / resamp_x, i_height / resamp_y))


    def get_output_details(self):
        return self.interpreters[-1].get_output_details()[0]


    def get_input_details(self):
        return self.interpreters[0].get_input_details()[0]

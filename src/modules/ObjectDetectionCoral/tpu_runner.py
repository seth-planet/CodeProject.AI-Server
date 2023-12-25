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
import multiprocessing
import queue
import platform
import sys

from datetime import datetime

import numpy as np
from PIL import Image

# For Linux we have installed the pycoral libs via apt-get, not PIP in the venv,
# So make sure the interpreters can find the coral libraries
if platform.system() == "Linux":
    sys.path.insert(0, "/usr/lib/python3.9/site-packages/")

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import list_edge_tpus
import pycoral.pipeline.pipelined_model_runner as pipeline
from pycoral.adapters import detect

from options import Options


INTERPRETER_LIFESPAN_SECONDS = 3600  # Refresh the interpreters once an hour

# Don't let the queues fill indefinitely until something more unexpected goes
# wrong. 1000 is arbitrarily chosen to block before things get ugly.
# It also implies that there are many threads calling into here and waiting on
# results. Our max queue lengths should never be more than
# calling_threads * tiles_per_image.
MAX_PIPELINE_QUEUE_LEN = 1000

# Warn if any TPU reads above this temperature C
# https://coral.ai/docs/pcie-parameters/#use-dynamic-frequency-scaling
WARN_TEMPERATURE_THRESHOLD = 80

# Nothing should ever sit in a queue longer than this many seconds.
# 60 seconds is arbitrarily chosen to throw an error eventually.
MAX_WAIT_TIME = 60.0


class TPURunner(object):

    def __init__(self):
        """
        Init object and do a check for the temperature file. Right now
        the temperature file would only be supported on Linux systems
        with the TPU installed on the PCIe bus.
        """
        # Refresh the interpreters once an hour
        self.interpreter_lifespan_secs = 3600

        self.interpreters        = []    # The model interpreters
        self.interpreter_created = None  # When were the interpreters created?
        self.labels              = None  # set of labels for this model
        self.runners             = None  # Pipeline(s) to run the model
        self.tpu_count           = 0     # Number of TPUs we end up using

        self.next_runner_idx     = 0     # Which runner to execute on next?
        self.postboxes           = None # Store output
        self.postmen             = None
        self.runner_lock         = threading.Lock()
        self.mp_pool             = None
        
        self.last_check_timer    = None

        logging.info("{} version: {}".format(Image.__name__, Image.__version__))

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
                    logging.info("Found temperature file at: "+fn.format(i))
                    return
        logging.debug("Unable to find a temperature file")
                    
                    
    def _post_service(self, r: pipeline.PipelinedModelRunner, q: queue.Queue):
        """
        A worker thread that loops to pull results from the pipelined model
        runner and deliver them to the requesting thread's queue. This is the
        main interface for receiving results from a pipeline runner.
        
        There are timeouts in the queues' usage here, but we don't expect them
        to ever be blocked. The intention is so if anything starts going 'off
        the rails' in regards to work syncronization, it blows up one of our
        queues first instead of giving us a more difficult OOM or deadlock
        condition.
        
        When a 'NOOP' is enqueued in the pipeline, it finishes its work and
        shuts down. That condition is detected here and exits the thread.
        """
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
            q.get(timeout=MAX_WAIT_TIME).put(rs, timeout=MAX_WAIT_TIME)


    def _get_devices():
        """Returns list of device names in usb:N or pci:N format.

        This function prefers returning PCI Edge TPU first.

        Returns:
        list of devices in pci:N and/or usb:N format

        Raises:
        RuntimeError: if not enough devices are available
        """
        edge_tpus = list_edge_tpus()

        num_pci_devices = sum(1 for device in edge_tpus if device['type'] == 'pci')
        logging.debug("{} PCIe TPUs detected".format(num_pci_devices))

        return ['pci:%d' % i for i in range(min(len(edge_tpus), num_pci_devices))] + [
          'usb:%d' % i for i in range(max(0, len(edge_tpus) - num_pci_devices))]


    # Should be called while holding runner_lock (if called at run time)
    def init_interpreters(self, options: Options) -> str:
        """
        Initializes the interpreters with the TFLite models.
        
        Also loads and initalizes the pipeline runners. To do this, it needs
        to figure out if we're using segmented pipelines, if we can load all
        the segments to the TPUs, and how to allocate them. For example, if
        we have three TPUs and request a model that contains two segments,
        we will load the two segments into two TPUs. If we have four TPUs
        and load the same model, we will create two pipeline runners with
        two segments each.
        
        We also kick off the 'postmen' here that are threads responsible for
        transferring output from the pipeline to queues held by each thread
        requesting infrence. These queues block until results are available
        for the thread. The postmen and queues are necessary because otherwise
        there would be no way of keeping track of which results align with
        which incoming request.
        
        The postmen each have 'postboxes' that are queues that they deliver
        results into. The next item in the postbox aligns with the next result
        in the pipeline. The items in the postboxes are themselves queues. This
        is because the thread requesting work is blocked waiting for a result
        to be placed in each of these queues (it's the postman's responability
        to make this transfer.) When the requesting thread's result is enqueued
        it is unblocked and proceeds to process the results.
        """

        self.interpreters = []
        self.runners = []
        
        segment_count = 1
        if any(options.tpu_segment_files):
            segment_count = max(1, len(options.tpu_segment_files))

        if self.mp_pool is None and options.resize_processes > 0:
            self.mp_pool = multiprocessing.Pool(processes=options.resize_processes)

        # Only allocate the TPUs we will use
        tpu_list = self._get_devices()
        self.tpu_count = (len(tpu_list) // segment_count) * segment_count
       
        # Read labels
        self.labels = read_label_file(options.label_file) \
                                                if options.label_file else {}
        
        # Initialize TF-Lite interpreters.
        device = ""
        try:
            device = "tpu"

            for i in range(self.tpu_count):
                # Alloc all segments into all TPUs, but no more than that.
                if segment_count > 1:
                    tpu_segment_file = \
                        options.tpu_segment_files[i % segment_count]
                else:
                    tpu_segment_file = options.model_tpu_file
                
                self.interpreters.append(make_interpreter(
                                            tpu_segment_file,
                                            device=tpu_list[i],
                                            delegate=None))
            logging.debug("Loaded {} TPUs".format(self.tpu_count))

            # Fallback to CPU
            if not any(self.interpreters):
                segment_count = 1
                device = "cpu"
                self.interpreters = [make_interpreter(
                                        options.model_cpu_file,
                                        device="cpu",
                                        delegate=None)]

        except Exception:
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
                self.interpreters = []
        
        # Womp womp
        if not any(self.interpreters):
            return ""

        # Initialize interpreters
        for i in self.interpreters:
            self.interpreters.allocate_tensors()

        self.interpreter_created = datetime.now()
        
        # Initialize runners/pipelines
        for i in range(0, self.tpu_count, segment_count):
            self.runners.append(
                pipeline.PipelinedModelRunner(
                    self.interpreters[i:i+segment_count]))
            
            self.runners.set_input_queue_size(MAX_PIPELINE_QUEUE_LEN)
            self.runners.set_output_queue_size(MAX_PIPELINE_QUEUE_LEN)

        # Setup postal queue
        self.postboxs = []
        self.postmen = []

        for r in self.runners:
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
        logging.debug("TPU & segment counts: {} & {}\n".format(self.tpu_count, segment_count))
        logging.debug("Interpreter count: {}\n".format(len(self.interpreters)))
        logging.debug(f"Input details: {input_details}\n")
        logging.debug(f"Output details: {output_details}\n")

        return device


    def _periodic_check(self, options: Options):
        """
        Run a periodic check to ensure the temeratures are good and we don't
        need to (re)initialize the interpreters/workers/pipelines. The system
        is setup to refresh the TF interpreters once an hour.
        
        I suspect that many of the problems I saw reported with the use of the
        Coral TPUs online were due to overheating chips. There were a few
        comments along the lines of: "Works great, but after running for a bit
        it became unstable and crashed. I had to back way off and it works fine
        now!" This seems symptomatic of the TPU throttling itself as it heats
        up, reducing its own workload, and giving unexpected results to the end
        user.
        """
        now_ts = datetime.now()
        
        # Check to make sure we aren't checking too often
        if self.last_check_timer != None and \
           (now_ts - self.last_check_timer).total_seconds() < 10:
            return
        self.last_check_timer = now_ts
        
        # Check temperatures
        msg = "Core {} is {} Celsius and will likely be throttled"
        if self.temp_fname_format != None:
            for i in len(self.interpreters):
                temp_arr = []
                if os.path.exists(self.temp_fname_format.format(i)):
                    with open(self.temp_fname_format.format(i), "r") as fp:
                        # Convert from milidegree C to degree C
                        temp = int(fp.read()) // 1000
                        temp_arr.append(temp)
                        
                        if WARN_TEMPERATURE_THRESHOLD <= temp:
                            logging.warning(msg.format(i, temp))
                logging.debug("Temperatures: {} avg; {} max; {} total".format(
                                                sum(temp_arr) // len(temp_arr),
                                                max(temp_arr),
                                                len(temp_arr)))

        # Once an hour, refresh the interpreters
        if any(self.interpreters):
            if (now_ts - self.interpreter_created).total_seconds() > \
                                                self.interpreter_lifespan_secs:
                logging.info("Refreshing the Tensorflow Interpreters")

                # Close all existing work before destroying...
                with self.runner_lock:
                    self._delete()
                    
                    # Re-init while we still have the lock
                    self.init_interpreters(options)

        # (Re)start them if needed
        if not any(self.interpreters):
            with self.runner_lock:
                self.init_interpreters(options)

        return any(self.interpreters)


    def _delete(self):
        """
        Close and delete each of the pipelines and interpreters while flushing
        existing work.
        """
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
        self.interpreters   = []


    def process_image(self,
                      options:Options,
                      image: Image,
                      score_threshold: float):
        """
        Execute all the default image processing operations.
        
        Will take an image and:
        - Initialize TPU pipelines.
        - Tile it.
        - Normalize each tile.
        - Run infrence on the tile.
        - Collate results.
        - Remove duplicate results.
        - Return results as Objects.
        - Return infrence timing.
        """

        if not self._periodic_check(options):
            return False

        all_objects = []
        all_queues = []
        name = self.get_input_details()['name']
        
        # Potentially resize & pipeline a number of tiles
        for rs_image, rs_loc in self._get_tiles(options, image):
            rs_queue = queue.Queue(maxsize=1)
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
        tot_infr_time = 0
        q_count = 0
        for rs_queue, rs_loc in all_queues:
            # Wait for results
            # We may have to wait a few seconds at most, but I'd expect the
            # pipeline to clear fairly quickly.
            start_inference_time = time.perf_counter()
            result = rs_queue.get(timeout=MAX_WAIT_TIME)
            tot_infr_time += time.perf_counter() - start_inference_time
            q_count += 1
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
                                                 bbox=bbox.map(int)))
        
        # Convert to ms
        tot_infr_time = int(tot_infr_time * 1000)

        if q_count <= 1:
            return all_objects, tot_infr_time

        # Remove duplicate objects
        idxs = self._non_max_suppression(all_objects, options.iou_threshold)
        
        return ([all_objects[i] for i in idxs], tot_infr_time)
        
        
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
        
        
    def _resize_and_chop_tiles(self,
                               options: Options,
                               image: Image,
                               queue: multiprocessing.SimpleQueue,
                               m_width,
                               m_height):
        """
        Run the image resizing in an independent process pool.
        
        Image resizing is one of the more expensive things we're doing here.
        It's expensive enough that it may take as much CPU time as inference
        under some circumstances. The Lanczos resampling kernel in particular
        is expensive, but results in quality output.
        
        For example, see the resizing performance charts here:
        https://python-pillow.org/pillow-perf
        
        Pillow is the highly optimized version of PIL and it only runs at
        ~100 MP/sec when making a thumbnail with the Lanczos kernel. That's
        only 12.6 4k frames per second, maximum, in a Python process. We are
        hoping to process more than that with TPU hardware.
        
        Besides multiprocessing, we can also improve performance by installing
        the 'pillow-simd' Python library. And improve it even more by
        re-compiling it to use AVX2 instructions. See:
        https://github.com/uploadcare/pillow-simd#pillow-simd
        """

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
                cropped_arr = resamp_img.crop((x_off,
                                               y_off,
                                               x_off + m_width,
                                               y_off + m_height)).asarray()
                logging.debug("Resampled image tile {}".format(cropped_arr.size))
            
                # Normalize input image
                normalized_input = zero_point + \
                    (cropped_arr.astype('float32') - cropped_arr.mean()) \
                                                (cropped_arr.std() * scale * 2)
                np.clip(normalized_input, 0, 255, out=normalized_input)

                # Print image stats
                logging.debug('Input Min: %.3f, Max: %.3f' %
                              (cropped_arr.min(), cropped_arr.max()))
                logging.debug('Normalized Min: %.3f, Max: %.3f' %
                              (normalized_input.min(), normalized_input.max()))
                logging.debug('Normalized Mean: %.3f, Standard Deviation: %.3f' %
                              (normalized_input.mean(), normalized_input.std()))

                queue.put((normalized_input.astype(np.uint8),
                          (x_off, y_off, i_width / resamp_x, i_height / resamp_y)))
        queue.put(None)
        return


    def _get_tiles(self, options: Options, image: Image):
        """
        Returns an iterator that yields image tiles and associated location.
        
        For tiling, we use the philosophy that it makes the most sense to
        keep the pixel downsampling multiplier somewhat constant and resample
        the image to fit multiples of the tensor input dimensions. The default
        option is a multiplier of roughly 6, which should give us two tiles for
        an image with HD or 4k dimensions. Anything larger or more
        square-shaped will be mapped to just be a single tile. This is
        intentionally kept conservative. To tile more agressively, reduce the
        multiplier down from 6. Run time will go up, as the number of
        inferences will go up with more tiles.
        
        If we don't tile the images, we end up with bad options:
        - We stretch the image to fit the tensor input. In the case of a 4k
        video stream, this basically doubles the height of images by streching
        them. Warping an image like this doesn't seem like it would improve AI
        performance.
        - We keep the aspect ratio the same and pad the image. In the case of
        a 4k image, this is wasting ~44% of the potential input data as simply
        padding. In the case of our smallest 300x300 model, a full 131x300
        pixels are wasted.
        
        It makes more sense to me to split the image in two; resulting in two
        tiles that are each neither very warped or have wasted input pixels.
        The downside is, of course, that we are doing twice as much work.
        
        Also normalizes each tile after it's chopped out. This is experimental
        and if it doesn't yield better results, should be dropped from the
        code. It would be particularly interesting to see how this affects
        night imagery.
        """
        _, m_height, m_width, _ = self.get_input_details()['shape']

        q = multiprocessing.SimpleQueue()
        if self.mp_pool == None:
            self._resize_and_chop_tiles(options, image, q, m_width, m_height)
        else:
            # Run resizing and tiling off the main thread. It'll take a while
            # and we don't want to block all threads with the GIL.
            self.mp_pool.apply_async(self._resize_and_chop_tiles,
                                     (options, image, q, m_width, m_height))

        # Return the tiles as they become available
        while True:
            rs = q.get()
            if rs == None:
                return
            yield rs


    def get_output_details(self):
        return self.interpreters[-1].get_output_details()[0]


    def get_input_details(self):
        return self.interpreters[0].get_input_details()[0]

# Lint as: python3
# Copyright 2019 Google LLC
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
r"""Example using PyCoral to classify a given image using an Edge TPU.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh classify_image.py

python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```

Running this directly from src\runtimes\bin\windows\python37:

cd \src\runtimes\bin\windows\python37
python.exe coral\pycoral\examples\classify_image.py --model coral\pycoral\test_data\mobilenet_v2_1.0_224_inat_bird_quant.tflite --labels coral\pycoral\test_data\inat_bird_labels.txt --input coral\pycoral\test_data\parrot.jpg
"""
import argparse
import time
import multiprocessing
import logging

from PIL import Image
from PIL import ImageDraw

from options import Options
from tpu_runner import TPURunner

from pycoral.utils.dataset import read_label_file

tpu_runner = TPURunner()


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True, nargs='+',
                      help='File path of .tflite file')
  parser.add_argument('-i', '--input', required=True,
                      help='File path of image to process')
  parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  parser.add_argument('-d', '--debug', action='store_true')
  args = parser.parse_args()

  if args.debug:
    logging.root.setLevel(logging.DEBUG)
  else:
    logging.root.setLevel(logging.INFO)

  options = Options()
  # Load segments
  if len(args.model) > 1:
    options.tpu_segment_names = args.model
  else:
    options.model_cpu_file = args.model[0]
    options.model_tpu_file = args.model[0]

  # Limit to one tile
  # Allows us apples-to-apples comparisons when benchmarking
  options.downsample_by  = 100
  
  # Don't use a pool of workers to resize & normalize images
  # FIXME: automate this option after further testing
  options.resize_processes = 0

  labels = read_label_file(args.labels) if args.labels else {}
  image = Image.open(args.input)

  print('----INFERENCE TIME----')
  print('Note: The first inference is slow because it includes',
        'loading the model into Edge TPU memory.')

  tot_infr_time = 0
  if args.count > 1:
    pool = multiprocessing.pool.ThreadPool(16)
    start = time.perf_counter()
    for _, infr_time in pool.starmap(tpu_runner.process_image,
            [(options, image, args.threshold) for i in range(args.count-1)]):
        tot_infr_time += infr_time
  else:
    start = time.perf_counter()

  objs, infr_time = tpu_runner.process_image(options, image, args.threshold)
  tot_infr_time += infr_time
  inference_time = time.perf_counter() - start
  print('%.2f ms avg wall time for each of %d runs' %
                            (inference_time * 1000 / args.count, args.count))
                            
  # Optimizing the number of segments used for a model would result in the
  # lowest average time spent adjusted for number of TPUs used. At some point,
  # adding additional segments just removes from the pool of TPUs you can use
  # for parallelism.
  print('%.2f ms avg time waiting for inference; %.2f avg TPU ms / run' %
                            (tot_infr_time / args.count,
                             tpu_runner.tpu_count * inference_time * 1000 / args.count))

  print('-------RESULTS--------')
  if not objs:
    print('No objects detected')
    return
  
  if any(objs):
    for obj in objs:
      print(labels.get(obj.id, obj.id))
      print('  id:    ', obj.id)
      print('  score: ', obj.score)
      print('  bbox:  ', obj.bbox)

  if args.output:
    image = image.convert('RGB')
    draw_objects(ImageDraw.Draw(image), objs, labels)
    image.save(args.output)
    image.show()

if __name__ == '__main__':
  main()

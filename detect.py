from yolov3.utils import parser
from yolov3.detect import detect


parser = parser.get_parser_from_arguments()
detect(parser)
